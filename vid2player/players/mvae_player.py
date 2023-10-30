from motion_vae.config import MotionVAEOption
from motion_vae.dataset import Video3DPoseDataset
from motion_vae.base import MotionVAEModel
from utils.racket import Racket
from utils.pose import SMPLPose
from utils.konia_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from utils.torch_transform import rot6d_to_angle_axis, rot6d_to_rotmat, rotmat_to_rot6d

from smpl_visualizer.smpl import SMPL, SMPL_MODEL_DIR

import math
import torch
from torch.utils.data import DataLoader


class MVAEPlayer:
    def __init__(self, cfg, num_envs, is_train, enable_physics, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self._is_train = is_train
        self._enable_physics = enable_physics
        self.device = device
        self._is_dual = cfg.get('dual_mode') == 'different'

        if not self._is_dual:
            # load mvae option
            opt = MotionVAEOption()
            opt.test_only = True
            opt.load(self.cfg['mvae_ver'])
            # load mvae model
            self._mvae = MotionVAEModel(opt, self.device)
        else:
            opt = MotionVAEOption()
            opt.test_only = True
            opt.load(self.cfg['mvae_ver'][0])
            self._mvae1 = MotionVAEModel(opt, self.device)
            opt2 = MotionVAEOption()
            opt2.test_only = True
            opt2.load(self.cfg['mvae_ver'][1])
            self._mvae2 = MotionVAEModel(opt2, self.device)

        self._root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._joint_rot6d = torch.zeros((self.num_envs, 24, 6), device=self.device, dtype=torch.float32)
        self._joint_rotmat = torch.zeros((self.num_envs, 24, 3, 3), device=self.device, dtype=torch.float32)
        self._racket_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._racket_normal = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._joint_rot = torch.zeros((self.num_envs, 24, 3), device=self.device, dtype=torch.float32)
        if not self._is_train:
            self._joint_rot_ori = torch.zeros((self.num_envs, 24, 3), device=self.device, dtype=torch.float32)
        
        self._predict_phase = self.cfg.get("predict_phase", True)
        if not self._is_dual:
            self._racket = Racket('tennis', self.device, righthand=self.cfg.get('righthand'))
        else:
            server = 0 if self.cfg.get('serve_from', 'near') != 'near' else 1
            self._racket = Racket('tennis', self.device, grip=self.cfg.get('grip')[server], righthand=self.cfg.get('righthand')[server])
        if self._predict_phase:
            self._phase_pred = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            self._swing_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64) - 1
            self._swing_type_cycle = self._swing_type.clone()

        # hard code gender as male        
        gender = 'male'
        self._smpl = SMPL(SMPL_MODEL_DIR, create_transl=False, 
            gender=gender,
            betas=self.cfg.get('smpl_beta', [[0] * 10]),
            batch_size=self.num_envs, device=self.device).to(self.device)

        self._court_min = torch.FloatTensor(self.cfg['court_min']).to(self.device)
        self._court_max = torch.FloatTensor(self.cfg['court_max']).to(self.device)
        self._court_range = self._court_max - self._court_min
        
        self._conditions = torch.zeros((self.num_envs, opt.num_condition_frames, opt.frame_size), 
            device=self.device, dtype=torch.float32)

        dim = 3
        if opt.condition_root_x_only: dim = 1
        if opt.no_condition_root_y: dim = 2
        if 'root_pos' not in opt.pose_feature: dim = 0
        if dim > 0:
            self._root_pos_inds = torch.LongTensor([i for i in range(dim)]).to(self.device)
        self._root_vel_inds = torch.LongTensor([i for i in range(dim, dim+3)]).to(self.device)
        self._joint_pos_inds = torch.LongTensor([i for i in range(dim+3, dim+3+23*3)]).to(self.device)
        dim += 3
        if 'joint_pos' in opt.pose_feature:
            dim += 23*3
        if 'joint_velo' in opt.pose_feature:
            dim += 23*3
        self._joint_rot6d_inds = torch.LongTensor([i for i in range(dim, dim + 24*6)]).to(self.device)
        
        if cfg.get('add_residual_dof'):
            if self._is_dual:
                self._residual_joint_inds = [None, None]
                for i, rh in enumerate(self.cfg['righthand']):
                    if rh: 
                        self._residual_joint_inds[i] = torch.LongTensor([
                            SMPLPose.RElbow, SMPLPose.RWrist]).to(self.device) 
                    else:
                        self._residual_joint_inds[i] = torch.LongTensor([
                            SMPLPose.LElbow, SMPLPose.LWrist]).to(self.device) 
            else:
                if cfg.get('righthand', True):
                    self._residual_joint_inds = torch.LongTensor([
                        SMPLPose.RElbow, SMPLPose.RWrist]).to(self.device)
                else:
                    self._residual_joint_inds = torch.LongTensor([
                        SMPLPose.LElbow, SMPLPose.LWrist]).to(self.device)
       
        self.init_states()
    
    def init_states(self):
        if self.cfg.get('dual_mode') == 'different': 
            init_data_fg = torch.load(self.cfg['vae_init_conditions_serve'][0])[:1].to(self.device).float()
            init_data_bg = torch.load(self.cfg['vae_init_conditions_serve'][1])[:1].to(self.device).float()
            self._init_data_serve = torch.cat([init_data_fg, init_data_bg], dim=0).repeat(self.num_envs // 2, 1, 1)

            init_data_fg = torch.load(self.cfg['vae_init_conditions_ready'][0])[:1].to(self.device).float()
            init_data_bg = torch.load(self.cfg['vae_init_conditions_ready'][1])[:1].to(self.device).float()
            self._init_data_ready = torch.cat([init_data_fg, init_data_bg], dim=0).repeat(self.num_envs // 2, 1, 1)
            return
        elif self.cfg.get('dual_mode') == True: 
            init_data_serve = torch.load(self.cfg['vae_init_conditions_serve'])[:1].to(self.device).float()
            self._init_data_serve = init_data_serve.repeat(self.num_envs, 1, 1)

            init_data_ready = torch.load(self.cfg['vae_init_conditions_ready'])[:1].to(self.device).float()
            self._init_data_ready = init_data_ready.repeat(self.num_envs, 1, 1)
            return

        # load mvae option
        opt = MotionVAEOption()
        opt.test_only = True
        opt.load(self.cfg['mvae_ver'])
        opt.nframes_seq = opt.num_condition_frames
        opt.batch_size = opt.nseqs = max(1000, self.num_envs)
        
        if self.cfg.get('vae_init_conditions') is not None:
            print('Load vae init conditions from', self.cfg['vae_init_conditions'])
            self._init_data = torch.load(self.cfg['vae_init_conditions']).to(self.device).float()
            if self._init_data.shape[0] < self.num_envs:
                self._init_data = self._init_data[:1].repeat(self.num_envs, 1, 1)
            elif self._init_data.shape[0] >= self.num_envs:
                self._init_data = self._init_data[:self.num_envs]
            return

        # load motion dataset
        self._dataset = Video3DPoseDataset(opt)
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(opt.num_threads))
        data_sampler = iter(self._data_loader)

        # sample init state for all envs
        batch_data = next(data_sampler)
        self._init_data = batch_data['feature'].to(self.device).float()

    def reset(self, env_ids):
        self._swing_type[env_ids] = -1
        self._swing_type_cycle[env_ids] = -1

        # assume VAE only conditioned on one frame
        self._update_mvae_state(self._init_data[env_ids, -1].clone(), phase=None, env_ids=env_ids, init=True)
        
    def reset_dual(self, reset_reaction_env_ids, reset_recovery_env_ids):
        env_ids = torch.cat([reset_reaction_env_ids, reset_recovery_env_ids]).sort().values
        self._swing_type[env_ids] = -1
        self._swing_type_cycle[env_ids] = -1

        self._conditions[reset_reaction_env_ids] = self._init_data_ready[reset_reaction_env_ids].clone()
        self._conditions[reset_recovery_env_ids] = self._init_data_serve[reset_recovery_env_ids].clone()

        if self._is_dual:
            serve_from = self.cfg.get('serve_from', 'near')
            self._update_mvae_state(self._conditions[env_ids[::2], -1], phase=None, env_ids=env_ids[::2], init=True,
                mvae=self._mvae1, residual_joint_inds=self._residual_joint_inds[0], serving=serve_from=='near')
            self._update_mvae_state(self._conditions[env_ids[1::2], -1], phase=None, env_ids=env_ids[1::2], init=True,
                mvae=self._mvae2, residual_joint_inds=self._residual_joint_inds[1], serving=serve_from=='far')
        else: 
            self._update_mvae_state(self._conditions[env_ids, -1], phase=None, env_ids=env_ids, init=True)

    def step(self, latents, residual=None):  
        self._latents = latents
        condition_flat = self._conditions.flatten(start_dim=1, end_dim=2)
        with torch.no_grad():
            if not self._is_dual:
                feature, phase = self._mvae.forward(latents, condition_flat)
                self._update_mvae_state(feature, phase, residual)
            else:
                feature1, phase1 = self._mvae1.forward(latents[::2], condition_flat[::2])
                feature2, phase2 = self._mvae2.forward(latents[1::2], condition_flat[1::2])
                self._update_mvae_state(feature1, phase1, residual, env_ids=torch.arange(0, self.num_envs, 2),
                    mvae=self._mvae1, player=self.cfg['player'][0],
                    residual_joint_inds=self._residual_joint_inds[0])
                self._update_mvae_state(feature2, phase2, residual, env_ids=torch.arange(1, self.num_envs, 2),
                    mvae=self._mvae2, player=self.cfg['player'][1],
                    residual_joint_inds=self._residual_joint_inds[1])

    def _update_mvae_state(self, feature, phase=None, residual=None, env_ids=None, init=False, mvae=None, 
        player=None, residual_joint_inds=None, serving=False):
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        if mvae is None:
            mvae = self._mvae
            player = self.cfg['player']
            residual_joint_inds = self._residual_joint_inds
        
        if init:
            feature_origin = feature.clone()
            if mvae.opt.condition_root_x_only:
                feature = torch.cat([feature[:, :1], feature[:, 3:]], dim=-1)
            if mvae.opt.no_condition_root_y:
                feature = torch.cat([feature[:, :1], feature[:, 2:]], dim=-1)
            if 'root_pos' not in mvae.opt.pose_feature:
                feature = feature[:, 3:]
            self._conditions[env_ids] = mvae.normalize(feature).view(len(env_ids), 1, -1)
        else:
            self._conditions[env_ids] = self._conditions[env_ids].roll(-1, dims=1)
            self._conditions[env_ids, -1] = feature # Normalized
            feature = mvae.unnormalize(feature)

        if init:
            self._root_pos[env_ids] = feature_origin[:, :3]

            new_root_xy = None
            if self._is_train or self.cfg.get('test_mode', 'random') == 'random' or self.cfg.get('dual_mode') is not None:
                # start player near court baseline center (default -1m to 1m)
                new_root_xy = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) \
                    * torch.FloatTensor([2, 1.5]).to(self.device) + torch.FloatTensor([0, -13]).to(self.device)
            else:
                # reset to center
                new_root_xy = torch.FloatTensor([0, -13]).repeat(len(env_ids), 1).to(self.device)

            if new_root_xy is not None:
                self._root_pos[env_ids, :2] = new_root_xy
                assert self._conditions.shape[1] == 1
                if 'root_pos' in mvae.opt.pose_feature:
                    new_conditions = self._conditions[env_ids].clone()
                    if mvae.opt.no_condition_root_y:
                        new_conditions[:, :, self._root_pos_inds] = mvae.normalize(
                            self._root_pos[env_ids][:, [0, 2]], self._root_pos_inds).view(len(env_ids), 1, 2)
                    elif mvae.opt.condition_root_x_only:
                        new_conditions[:, :, self._root_pos_inds] = mvae.normalize(
                            self._root_pos[env_ids][:, [0]], self._root_pos_inds).view(len(env_ids), 1, 1)
                    else:
                        new_conditions[:, :, self._root_pos_inds] = mvae.normalize(
                            self._root_pos[env_ids], self._root_pos_inds).view(len(env_ids), 1, 3)
                    self._conditions[env_ids] = new_conditions
        else:
            self._root_pos[env_ids] = self._root_pos[env_ids] + feature[:, self._root_vel_inds]
            # replace root position(normalized) in the condition
            if 'root_pos' in mvae.opt.pose_feature:
                if mvae.opt.no_condition_root_y:
                    self._conditions[:, -1, self._root_pos_inds] = mvae.normalize(self._root_pos[:, [0, 2]], self._root_pos_inds)
                elif mvae.opt.condition_root_x_only:
                    self._conditions[:, -1, self._root_pos_inds] = mvae.normalize(self._root_pos[:, [0]], self._root_pos_inds)
                else:
                    self._conditions[:, -1, self._root_pos_inds] = mvae.normalize(self._root_pos, self._root_pos_inds)
        self._root_vel[env_ids] = feature[:, self._root_vel_inds]
        
        joint_rot6d = feature[:, self._joint_rot6d_inds].view(-1, 24, 6)
        self._joint_rot6d[env_ids] = joint_rot6d
        self._joint_rotmat[env_ids] = rot6d_to_rotmat(joint_rot6d)
        if self.cfg.get('vis_mvae') == 'origin' and not self._is_train:
            self._joint_rot_ori[env_ids] = rot6d_to_angle_axis(self._joint_rot6d[env_ids])
        # kinematic joint pos predicted by VAE
        joint_pos = feature[:, self._joint_pos_inds].view(-1, 23, 3)

        # setup phase and swing type
        if self._predict_phase and phase is not None:
            self._phase_pred[env_ids] = torch.atan2(phase[:, 0], phase[:, 1])
            self._phase_pred[env_ids] = torch.where(self._phase_pred[env_ids] < 0, 
                self._phase_pred[env_ids] + math.pi * 2, self._phase_pred[env_ids])
            if self._is_dual:
                righthand = self.cfg['righthand'][env_ids[0].item() % 2]
            else:
                righthand = self.cfg.get('righthand', True)
            self._swing_type[env_ids] = torch.where(
                (self._swing_type[env_ids] == -1) & (self._phase_pred[env_ids] > 2.0) & (self._phase_pred[env_ids] < 3.5),
                torch.where(joint_pos[:, SMPLPose.RWrist-1, 0] > 0,
                    torch.ones_like(self._swing_type[env_ids]) * 1 if righthand else 2, 
                    torch.ones_like(self._swing_type[env_ids]) * 2 if righthand else 1),
                self._swing_type[env_ids]
            )
            # reset to unknown after 3.5
            self._swing_type[env_ids] = torch.where(
                (self._swing_type[env_ids] != -1) & (self._phase_pred[env_ids] > 3.5),
                torch.zeros_like(self._swing_type[env_ids]) - 1,
                self._swing_type[env_ids]
            )
            self._swing_type_cycle[env_ids] = torch.where(
                self._swing_type[env_ids] != -1,
                self._swing_type[env_ids],
                self._swing_type_cycle[env_ids]
            )
        
        # add residual angles
        if residual is not None and len(residual) != 0:
            assert self.cfg.get('add_residual_dof') == 'euler'
            residual = residual.clone()

            if player == 'djokovic':
                residual[:, 0] = torch.clamp(residual[:, 0] * 0.25, -0.25, 0.25)
                residual[:, 1] = torch.clamp(residual[:, 1] * 0.25, -0.25, 0.25)
                residual[:, 2] = torch.clamp(residual[:, 2] * 0.25, -0.25, 0.25)

                joint_rot = rotation_matrix_to_angle_axis(self._joint_rotmat[:, residual_joint_inds])
                during_fh_swing = (self._phase_pred > 2.0) & (self._phase_pred < 3.2) & (self._swing_type == 1)
                during_fh_pre_contact = (self._phase_pred > 2.0) & (self._phase_pred < 3.1) & (self._swing_type == 1)
                during_fh_post_contact = (self._phase_pred > 2.0) & (self._phase_pred >= 3.1) & (self._phase_pred < 3.2) & (self._swing_type == 1)

                during_bh_swing = (self._phase_pred > 2.0) & (self._phase_pred < 3.2) & (self._swing_type == 2)
                during_bh_pre_contact = (self._phase_pred > 2.0) & (self._phase_pred < 3.0) & (self._swing_type == 2)

                eblow_twist_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                wrist_swing_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                eblow_twist_base[during_fh_swing] = -0.75
                wrist_swing_base[during_fh_pre_contact] = -0.25
                wrist_swing_base[during_fh_post_contact] = 0.25

                eblow_twist_base[during_bh_swing] = -0.25
                wrist_swing_base[during_bh_pre_contact] = 0.1

                if not self._is_train:
                    not_swing = ~(during_fh_swing | during_bh_swing)
                    residual[not_swing, :] = 0

                joint_rot[:, 0, 0] = (eblow_twist_base + residual[:, 0]) * math.pi # elbow twist
                joint_rot[:, 1, 0] = 0 # wrist twist = 0
                joint_rot[:, 1, 1] = residual[:, 1] * math.pi # wrist shake 
                joint_rot[:, 1, 2] = (wrist_swing_base + residual[:, 2]) * math.pi # wrist swing 

            elif player == 'federer':
                residual[:, 0] = torch.clamp(residual[:, 0] * 0.25, -0.25, 0.25)
                residual[:, 1] = torch.clamp(residual[:, 1] * 0.25, -0.25, 0.25)
                residual[:, 2] = torch.clamp(residual[:, 2] * 0.25, -0.25, 0.25)

                joint_rot = rotation_matrix_to_angle_axis(self._joint_rotmat[:, residual_joint_inds])
                during_fh_swing = (self._phase_pred > 2.0) & (self._phase_pred < 3.2) & (self._swing_type == 1)
                during_fh_pre_contact = (self._phase_pred > 2.0) & (self._phase_pred < 3.1) & (self._swing_type == 1)
                during_fh_post_contact = (self._phase_pred > 2.0) & (self._phase_pred >= 3.1) & (self._phase_pred < 3.2) & (self._swing_type == 1)
                during_bh_swing = (self._phase_pred > 2.0) & (self._phase_pred < 3.5) & (self._swing_type == 2)
                during_bh_pre_contact = (self._phase_pred > 2.0) & (self._phase_pred < 3.0) & (self._swing_type == 2)
                during_bh_slice_swing = (self._phase_pred > 2.0) & (self._phase_pred < 3.5) & (self._swing_type == 3)
                during_serve = (self._phase_pred > 2.0) & (self._phase_pred < 3.3) & (self._swing_type == 0)
                pre_serve = (self._phase_pred > 2.0) & (self._phase_pred < 3.0) & (self._swing_type == 0)

                wrist_twist_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                wrist_shake_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                wrist_swing_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                eblow_twist_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

                eblow_twist_base[during_fh_swing] = -0.5
                wrist_swing_base[during_fh_pre_contact] = -0.25
                wrist_swing_base[during_fh_post_contact] = 0.25

                wrist_twist_base[during_bh_swing] = -0.25 # for changing grip
                wrist_shake_base[during_bh_swing] = 0.15

                wrist_twist_base[during_bh_slice_swing] = -0.1 # for changing grip

                wrist_twist_base[during_serve] = -0.5 # for changing grip
                wrist_shake_base[during_serve] = 0.1
                eblow_twist_base[during_serve] = -0.25
                wrist_swing_base[pre_serve] = -0.5

                if not self._is_train:
                    not_swing = ~(during_fh_swing | during_bh_swing)
                    residual[not_swing, :] = 0

                joint_rot[:, 0, 0] = (eblow_twist_base + residual[:, 0]) * math.pi # elbow twist
                joint_rot[:, 1, 0] = wrist_twist_base * math.pi
                joint_rot[:, 1, 1] = (wrist_shake_base + residual[:, 1]) * math.pi # wrist shake 
                joint_rot[:, 1, 2] = (wrist_swing_base + residual[:, 2]) * math.pi # wrist swing 

            elif player == 'nadal':
                residual[:, 0] = torch.clamp(residual[:, 0] * 0.25, -0.25, 0.25)
                residual[:, 1] = torch.clamp(residual[:, 1] * 0.25, -0.25, 0.25)
                residual[:, 2] = torch.clamp(residual[:, 2] * 0.25, -0.25, 0.25)

                joint_rot = rotation_matrix_to_angle_axis(self._joint_rotmat[:, residual_joint_inds])
                during_fh_swing = (self._phase_pred > 2.5) & (self._phase_pred < 3.2) & (self._swing_type == 1)
                during_bh_swing = (self._phase_pred > 2.0) & (self._phase_pred < 3.5) & (self._swing_type == 2)
                during_bh_pre_contact = (self._phase_pred > 2.0) & (self._phase_pred < 3.0) & (self._swing_type == 2)

                wrist_twist_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                wrist_shake_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                wrist_swing_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                eblow_twist_base = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

                eblow_twist_base[during_fh_swing] = -0.75
                wrist_swing_base[during_fh_swing] = 0.25

                wrist_twist_base[during_bh_swing] = -0.4 # for changing grip
                wrist_swing_base[during_bh_pre_contact] = -0.25

                if not self._is_train:
                    not_swing = ~(during_fh_swing | during_bh_swing)
                    residual[not_swing, :] = 0

                joint_rot[:, 0, 0] = (eblow_twist_base + residual[:, 0]) * math.pi # elbow twist
                joint_rot[:, 1, 0] = wrist_twist_base * math.pi
                joint_rot[:, 1, 1] = (wrist_shake_base + residual[:, 1]) * math.pi # wrist shake 
                joint_rot[:, 1, 2] = (wrist_swing_base + residual[:, 2]) * math.pi # wrist swing 

            if self._is_dual:
                joint_rotmat = self._joint_rotmat[env_ids].clone()
                joint_rotmat[:, residual_joint_inds] = angle_axis_to_rotation_matrix(joint_rot[env_ids])
                self._joint_rotmat[env_ids] = joint_rotmat
                joint_rot6d = self._joint_rot6d[env_ids].clone()
                joint_rot6d[:, residual_joint_inds] = rotmat_to_rot6d(joint_rotmat[:, residual_joint_inds])
                self._joint_rot6d[env_ids] = joint_rot6d
            else:
                self._joint_rotmat[:, residual_joint_inds] = angle_axis_to_rotation_matrix(joint_rot)
                self._joint_rot6d[:, residual_joint_inds] = rotmat_to_rot6d(self._joint_rotmat[:, residual_joint_inds])

        if not self._is_train:
            self._joint_rot[env_ids] = rot6d_to_angle_axis(self._joint_rot6d[env_ids])

        if init:
            racket = self._racket.infer_with_fk(
                joint_rotmat=self._joint_rotmat[env_ids],
                joint_pos_bind_rel=self._smpl.joint_pos_bind_rel,
                root_pos=self._root_pos[env_ids],
            )
            self._racket_pos[env_ids] = racket['pos']
            self._racket_normal[env_ids] = racket['normal']