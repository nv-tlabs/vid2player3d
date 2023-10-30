from motion_vae.base import MotionVAEModel
from motion_vae.dataset import Video3DPoseDataset, encode_action
from utils.common import *
from utils.racket import infer_racket_from_smpl

from smpl_visualizer.vis_sport import SportVisualizer
from smpl_visualizer.vis import vstack_videos
from smpl_visualizer.smpl import SMPL, SMPL_MODEL_DIR

import torch
import copy
import os
from tqdm import tqdm


class BaseRunner(object):

    def __init__(self):
        self.root_cur = None
        self.joint_pos_cur = None
        self.root_history = []
        self.joint_pos_history = []


    def step(self):
        pass


class MotionVAERunner(BaseRunner):

    def __init__(self, opt):
        super().__init__()

        opt = copy.deepcopy(opt)
        opt.test_only = True
        self.motion_vae = MotionVAEModel(opt)

        self.base_action = torch.zeros(opt.latent_size).float()
        self.base_action.normal_(0, 1)
        self.latent = None
        self.action = None
        self.phase = torch.FloatTensor([0, 1])
        self.phase_rad = 0

    def init_state(self, dataset):
        opt = self.motion_vae.opt
        first_frame = dataset.sample_first_frame()
        self.root_cur = torch.from_numpy(first_frame['root_pos']).float()
        if opt.update_joint_pos:
            self.joint_pos_cur = torch.from_numpy(first_frame['joint_pos']).float()
        else:
            self.joint_rot_cur = torch.from_numpy(first_frame['joint_rot']).float()
        self.condition = torch.from_numpy(first_frame['condition']).float() # T x F
        
        # initialize root
        self.root_cur[:2] = torch.FloatTensor([0, -12])

        self.root_history = self.root_cur.unsqueeze(0).clone()
        if opt.update_joint_pos:
            self.joint_pos_history = self.joint_pos_cur.view(23, 3).unsqueeze(0).clone()
        else:
            self.joint_rot_history = self.joint_rot_cur.view(24, 3).unsqueeze(0).clone()


    def set_latent_random(self):
        latent = torch.zeros_like(self.base_action)
        latent.normal_(0, 1)
        self.latent = latent
    

    def set_action(self):
        action = torch.LongTensor([6])
        self.action = encode_action(action, self.motion_vae.opt.action_dim)


    def step(self):
        next_frame = self.motion_vae.infer_single(self.latent, self.condition, self.action)
        self.update_state(next_frame)
    

    def update_state(self, frame):
        opt = self.motion_vae.opt
        self.root_cur = self.root_cur + frame['root_velo']
        # bring player back to court
        court_bbox = torch.FloatTensor([-5, -15, 5, 0])
        if not test_point_in_bbox(self.root_cur[:2], court_bbox):
            self.root_cur[:2] = torch.FloatTensor([0, -13])

        self.root_history = torch.cat((self.root_history, self.root_cur.unsqueeze(0)))
        if opt.update_joint_pos:
            self.joint_pos_cur = frame['joint_pos']
            self.joint_pos_history = torch.cat((self.joint_pos_history, self.joint_pos_cur.view(23, 3).unsqueeze(0)))
        else:
            self.joint_rot_cur = frame['joint_rot']
            self.joint_rot_history = torch.cat((self.joint_rot_history, self.joint_rot_cur.view(24, 3).unsqueeze(0)))
        self.condition = self.condition.roll(-1, dims=0)
        self.condition[-1].copy_(frame['feature'])
        if 'root_pos' in opt.pose_feature:
            self.condition[-1, :3].copy_(self.root_cur)
        if opt.predict_phase:
            self.phase = frame['phase']
            self.phase_rad = frame['phase_rad']
        

def test_motion_vae_randomwalk(opt, num_test=5, num_runner=5, result_dir_suffix='',
    same_init_state=True, nframes=1000, interactive=False, 
    ):
    """
    random walk for motion vae model
    """
    result_dir = os.path.join(opt.result_dir, opt.model_ver + result_dir_suffix)
    print("Save video results to {}".format(result_dir))
    
    visualizer = SportVisualizer(
        verbose=False, 
        show_smpl=not opt.update_joint_pos,
        show_skeleton=False,
        show_racket=opt.infer_racket,
        correct_root_height=True,
        gender='male',
    )
    opt.batch_size = 1e9 # HACK for random sampling
    dataset = Video3DPoseDataset(opt)

    if opt.infer_racket:
        smpl = SMPL(SMPL_MODEL_DIR, create_transl=False, gender='male')

    # render a video for each clip, start with the initial frame of the clip
    for tid in range(num_test):
        tid += 1
        result_sub_dir = os.path.join(result_dir, '{:03}'.format(tid))
        os.makedirs(result_sub_dir, exist_ok=True)
        print("Running test", tid)

        runner_dict = {}
        for r in range(num_runner):
            set_seed(tid if same_init_state else tid + r)
            runner_dict[r] = MotionVAERunner(opt)
            runner_dict[r].init_state(dataset)

        for idx in tqdm(range(nframes - 1)):
            for r in range(num_runner):
                runner_dict[r].set_latent_random()
                runner_dict[r].step()
        
        # render video
        joint_pos_all = torch.zeros((num_runner, nframes, 24, 3))
        joint_rot_all = torch.zeros((num_runner, nframes, 24, 3))
        trans_all = torch.empty((num_runner, nframes, 3))
        for r in range(num_runner):
            if opt.update_joint_pos:
                joint_pos_all[r, :, 1:, :] = runner_dict[r].joint_pos_history # N x 23 x 3
            else:
                joint_rot_all[r, :, :, :] = runner_dict[r].joint_rot_history # N x 24 x 3
            trans_all[r, ...] = runner_dict[r].root_history # N x 3
        joint_rot_all[..., -1, :] = torch.FloatTensor([0, 0, np.pi/2])
        if opt.infer_racket:
            smpl_motion = smpl(
                global_orient=joint_rot_all[:, :, 0].reshape(-1, 3),
                body_pose=joint_rot_all[:, :, 1:].reshape(-1, 69),
                betas=torch.zeros(num_runner*nframes, 10).float(),
                root_trans = trans_all.reshape(-1, 3),
                return_full_pose=True,
                orig_joints=True
            )
            joint_pos_all = smpl_motion.joints.reshape(num_runner, nframes, 24, 3) - \
                trans_all.reshape(num_runner, nframes, 1, 3)
            racket_all = []
            for r in range(num_runner):
                racket_all.append([])
                for i in range(nframes):
                    racket_all[r] += [infer_racket_from_smpl(
                        joint_pos_all[r][i].numpy(), joint_rot_all[r][i].numpy(), 
                        sport=opt.sport, righthand=opt.player_name!=['Nadal'])]
        
        smpl_seq = {
            'trans': trans_all,
            'orient': None,
            'betas': torch.zeros((num_runner, 10)),
        }
        if opt.update_joint_pos:
            smpl_seq['joint_pos'] = joint_pos_all
        else:
            smpl_seq['joint_rot'] = joint_rot_all.view(num_runner, nframes, 24*3)
        
        init_args = {
            'smpl_seq': smpl_seq, 
            'num_actors': num_runner, 
            'sport': opt.sport,
            'camera': 'front',
            'racket_seq': racket_all if opt.infer_racket else None,
        }
        vid_path = os.path.join(result_sub_dir, 'random_front.mp4') 
        if interactive:
            visualizer.show_animation(
                init_args=init_args, 
                fps=30, 
                window_size=(1000, 1000), 
                enable_shadow=True
            )
        else:
            visualizer.save_animation_as_video(
                vid_path,
                init_args=init_args, 
                fps=30, 
                window_size=(1000, 1000), 
                enable_shadow=True,
                cleanup=True
            )