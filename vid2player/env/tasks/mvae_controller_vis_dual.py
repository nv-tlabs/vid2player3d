from env.tasks.physics_mvae_controller_dual import PhysicsMVAEControllerDual
from utils.racket import infer_racket_from_smpl
from utils.torch_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

from smpl_visualizer.vis_sport import SportVisualizer
from smpl_visualizer.vis_scenepic import SportVisualizerHTML
from smpl_visualizer.vis import images_to_video

import os
import torch
import tempfile
import shutil
from tqdm import tqdm


class MVAEControllerVisDual(PhysicsMVAEControllerDual):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params, 
                         physics_engine=physics_engine, 
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.headless = headless
        self.frame_index = 0
        self.num_best = self.cfg['env'].get('num_eg', 1)
        
        if self.cfg['env'].get('record_scenepic'):
            self.html_visualizer = SportVisualizerHTML(self._smpl, show_ball=True)
            self.smpl_verts = None
            self.racket_params = [[] for _ in range(self.num_envs)]
            self.ball_params = [[] for _ in range(self.num_envs)]
        elif self.cfg['env'].get('record') or not self.headless:
            init_args = {
                'num_actors': 2, 
                'sport': 'tennis',
                'camera': self.cfg['env'].get('camera', 'front'),
            }
            # window size has to be set as 1000x1000 due to bug in vtk
            self.visualizer = SportVisualizer(verbose=False, 
                show_smpl=True, show_skeleton=True, show_racket=True,
                show_target=False, show_ball=True, show_ball_target=False,
                enable_shadow=True)
            self.visualizer.show_animation_online(
                init_args=init_args,
                window_size=(1000, 1000),
                off_screen=self.headless,
            )

        # hardcode maximum length       
        episode_length = 1800
        self.joint_rot_all = torch.zeros((self.num_envs, episode_length, 24, 3), dtype=torch.float32)
        self.root_pos_all = torch.zeros((self.num_envs, episode_length, 3), dtype=torch.float32)
        self.ball_pos_all = torch.zeros((self.num_envs, episode_length, 3), dtype=torch.float32)
        self.target_pos_all = torch.zeros((self.num_envs, episode_length, 3), dtype=torch.float32)
        self.phase_all = torch.zeros((self.num_envs, episode_length), dtype=torch.float32)
        self.swing_type_all = torch.zeros((self.num_envs, episode_length), dtype=torch.int64)
        self.tar_action_all = torch.zeros((self.num_envs, episode_length), dtype=torch.int64)
        self.distance_best_all = []
        self.joint_rot_best_all = []
        self.root_pos_best_all = []
        self.ball_pos_best_all = []
        self.phase_best_all = []
        self.swing_type_best_all = []
        self.tar_action_best_all = []
    
    def start_recording(self, eg_id):
        camera = self.cfg['env'].get('camera', 'front')
        self.record_path = self.cfg['env'].get('rec_fname', 
            f"out/video/{self.cfg['args'].cfg}_{eg_id:03d}_{camera}.mp4")
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)

        self.frame_index = 0
        self.frame_dir = tempfile.mkdtemp(prefix="mvae_controller_vis-")
        if os.path.exists(self.frame_dir):
            shutil.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir, exist_ok=True)
        print(f"============ Writing video to {self.record_path} ============")

    def end_recording(self):
        images_to_video(self.frame_dir, self.record_path, fps=30, crf=25, verbose=False)
        shutil.rmtree(self.frame_dir)
        print(f"============ Video finished writing ============")
    
    def render_vis(self, init=False):
        self.frame_index += 1
        if not self.cfg['env'].get('record') and not self.cfg['env'].get('record_scenepic'):
            if self.headless: return
            
            smpl_motion = self._smpl(
                    global_orient=self._joint_rot[:2, 0].reshape(-1, 3),
                    body_pose=self._joint_rot[:2, 1:].reshape(-1, 69),
                    betas=self.betas[:2],
                    root_trans = self._root_pos[:2].reshape(-1, 3),
                    return_full_pose=True,
                    orig_joints=True
                )
            
            smpl_verts = smpl_motion.vertices.reshape(2, -1, 3)
            joint_pos = smpl_motion.joints.reshape(2, 24, 3)        

            joint_pos_rel = joint_pos - self._root_pos[:2].view(-1, 1, 3)
            racket_params = []
            for i in range(2):
                if self.cfg_v2p.get('dual_mode') == 'different':
                    grip = self.cfg_v2p['grip'][i]
                    righthand = self.cfg_v2p['righthand'][i]
                else:
                    grip = self.cfg_v2p.get('grip', 'eastern')
                    righthand = self.cfg_v2p.get('righthand', True)
                racket_params += [infer_racket_from_smpl(
                    joint_pos_rel[i].cpu().numpy(), 
                    self._joint_rot[i].cpu().numpy(), 
                    sport='tennis', grip=grip, righthand=righthand)]
                racket_params[-1]['root'] = self._root_pos[i].cpu().numpy()

            # flip far-side player
            smpl_verts[1, :, :2] *= -1
            joint_pos[1, :, :2] *= -1
            racket_params[1]['racket_dir'][:2] *= -1
            racket_params[1]['head_center'][:2] *= -1
            racket_params[1]['racket_normal'][:2] *= -1
            racket_params[1]['shaft_left_center'][:2] *= -1
            racket_params[1]['shaft_right_center'][:2] *= -1
            racket_params[1]['shaft_left_dir'][:2] *= -1
            racket_params[1]['shaft_right_dir'][:2] *= -1
            racket_params[1]['handle_center'][:2] *= -1
            racket_params[1]['root'][:2] *= -1

            # flip ball and target
            env_id = 0 if self._tar_action[0] == 1 else 1
            ball_params = [{
                'pos': self._ball_pos[env_id].clone(),
                # 'ang_vel': F.normalize(self._physics_player.task._ball_root_states[env_id, 10:13], dim=0),
            }]
            ball_targets = self._target_bounce_pos[[env_id]]
            if env_id == 1:
                ball_params[0]['pos'][:2] *= -1
                ball_targets[:, :2] *= -1
            if self.cfg_v2p.get('dual_mode') == 'different':
                righthand = self.cfg_v2p['righthand'][env_id]
            else:
                righthand = self.cfg_v2p.get('righthand', True)

            self.visualizer.update_scene_online(
                smpl_verts=smpl_verts, 
                joint_pos=joint_pos, 
                racket_params=racket_params,
                ball_params=ball_params,
                ball_targets=ball_targets,
            )
            self.visualizer.render_online(interactive=not self.headless)
            return

        env_ids = torch.arange(self.num_envs)
        self.joint_rot_all[env_ids, self.progress_buf] = self._joint_rot.cpu()
        self.root_pos_all[env_ids, self.progress_buf] = self._root_pos.cpu()
        self.ball_pos_all[env_ids, self.progress_buf] = self._ball_pos.cpu()
        self.phase_all[env_ids, self.progress_buf] = self._mvae_player._phase_pred.cpu()
        self.swing_type_all[env_ids, self.progress_buf] = self._mvae_player._swing_type_cycle.cpu()
        self.tar_action_all[env_ids, self.progress_buf] = self._tar_action.cpu()

        if self.reset_buf.sum() > 0:
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            # select max distance as best
            distance = self._distance[reset_env_ids[::2]] + self._distance[reset_env_ids[1::2]]
            max_dist = distance.max()
            if len(self.distance_best_all) == 0 or max_dist > self.distance_best_all[-1]:
                best_env_id = reset_env_ids[::2][torch.argmax(distance)]
                nframes = self.progress_buf[best_env_id]
                assert best_env_id % 2 == 0
                best_env_ids = torch.LongTensor([best_env_id, best_env_id+1])
               
                insert_id = 0
                for id, dist in enumerate(self.distance_best_all):
                    if max_dist > dist: break
                    insert_id += 1
                self.distance_best_all.insert(insert_id, max_dist)
                self.joint_rot_best_all.insert(insert_id, self.joint_rot_all[best_env_ids, :nframes].clone())
                self.root_pos_best_all.insert(insert_id, self.root_pos_all[best_env_ids, :nframes].clone())
                self.ball_pos_best_all.insert(insert_id, self.ball_pos_all[best_env_ids, :nframes].clone())
                self.phase_best_all.insert(insert_id, self.phase_all[best_env_ids, :nframes].clone())
                self.swing_type_best_all.insert(insert_id, self.swing_type_all[best_env_ids, :nframes].clone())
                self.tar_action_best_all.insert(insert_id, self.tar_action_all[best_env_ids, :nframes].clone())

                print("Update max episode length {}, distance {} from {}".format(nframes, max_dist, best_env_ids))
        # only store the top N
        self.distance_best_all = self.distance_best_all[:self.num_best]
        self.joint_rot_best_all = self.joint_rot_best_all[:self.num_best]
        self.root_pos_best_all = self.root_pos_best_all[:self.num_best]
        self.ball_pos_best_all = self.ball_pos_best_all[:self.num_best]
        self.phase_best_all = self.phase_best_all[:self.num_best]
        self.swing_type_best_all = self.swing_type_best_all[:self.num_best]
        self.tar_action_best_all = self.tar_action_best_all[:self.num_best]

        if self.frame_index >= self.cfg['env'].get('num_rec_frames', 300):
            # save results        
            start_eg_id = self.cfg['env'].get('start_eg_id', 1)
            for i in range(self.num_best):
                self.render_one_result(i, start_eg_id+i)
            exit()

    def render_one_result(self, best_idx, eg_id):
        if self.cfg['env'].get('record'):
            self.start_recording(eg_id)

        joint_rot = self.joint_rot_best_all[best_idx].to(self.device)
        root_pos = self.root_pos_best_all[best_idx].to(self.device)
        ball_pos = self.ball_pos_best_all[best_idx].to(self.device)
        tar_action = self.tar_action_best_all[best_idx].to(self.device)

        if self.cfg_v2p.get('fix_two_hand_backhand_post'):
            phase = self.phase_best_all[best_idx]
            swing_type = self.swing_type_best_all[best_idx]

            for j in range(2):
                if self.cfg_v2p.get('dual_mode') == 'different':
                    player = self.cfg_v2p['player'][j]
                    righthand = self.cfg_v2p['righthand'][j]
                else:
                    player = self.cfg_v2p.get('player')
                    righthand = self.cfg_v2p.get('righthand', True)
                if player.startswith('federer'): continue

                need_fix = (swing_type[j] == 2) & (phase[j] > 2.0) & (phase[j] < 5.0)
                joint_rotmat = angle_axis_to_rotation_matrix(joint_rot[j, need_fix])

                joint_rotmat = self._physics_player.task.optimize_two_hand_backhand(
                    joint_rotmat, single=True, righthand=righthand)
                joint_rot[j, need_fix] = rotation_matrix_to_angle_axis(joint_rotmat)

        nframes = root_pos.shape[1]
        smpl_motion = self._smpl(
            global_orient=joint_rot[:, :, 0].reshape(-1, 3),
            body_pose=joint_rot[:, :, 1:].reshape(-1, 69),
            betas=self.betas[:2].view(2, 1, 10).repeat(1, nframes, 1).reshape(-1, 10).to(self.device),
            root_trans = root_pos.reshape(-1, 3),
            return_full_pose=True,
            orig_joints=True
        )
    
        smpl_verts = smpl_motion.vertices.reshape(2, nframes, -1, 3)
        joint_pos = smpl_motion.joints.reshape(2, nframes, 24, 3)
        joint_pos_rel = joint_pos - root_pos.view(2, nframes, 1, 3)

        racket_params = []
        for i in range(nframes):
            params = []
            for j in range(2):
                if self.cfg_v2p.get('dual_mode') == 'different':
                    grip = self.cfg_v2p['grip'][j]
                    righthand = self.cfg_v2p['righthand'][j]
                else:
                    grip = self.cfg_v2p.get('grip', 'eastern')
                    righthand = self.cfg_v2p.get('righthand', True)
                params += [infer_racket_from_smpl(
                    joint_pos_rel[j, i].cpu().numpy(), 
                    joint_rot[j, i].cpu().numpy(), 
                    sport='tennis', grip=grip, righthand=righthand)]
                params[-1]['root'] = root_pos[j, i].cpu().numpy()
            racket_params += [params]
        
        # flip far-side player
        smpl_verts[1, :, :, :2] *= -1
        joint_pos[1, :, :, :2] *= -1
        for i in range(nframes):
            racket_params[i][1]['racket_dir'][:2] *= -1
            racket_params[i][1]['head_center'][:2] *= -1
            racket_params[i][1]['racket_normal'][:2] *= -1
            racket_params[i][1]['shaft_left_center'][:2] *= -1
            racket_params[i][1]['shaft_right_center'][:2] *= -1
            racket_params[i][1]['shaft_left_dir'][:2] *= -1
            racket_params[i][1]['shaft_right_dir'][:2] *= -1
            racket_params[i][1]['handle_center'][:2] *= -1
            racket_params[i][1]['root'][:2] *= -1
                    
        ball_params = []
        for i in range(nframes):
            if tar_action[0, i] == 1:
                pos = ball_pos[0, i]
            else:
                pos = ball_pos[1, i]
                pos[:2] *= -1
            ball_params += [{'pos': pos}]

        print(f"Rendering motion from result {best_idx} ...")
        if self.cfg['env'].get('record') or not self.headless:
            for i in tqdm(range(nframes)):
                self.visualizer.update_scene_online(
                    smpl_verts=smpl_verts[:, i], 
                    joint_pos=joint_pos[:, i], 
                    racket_params=racket_params[i],
                    ball_params=ball_params[i:i+1],
                )
                self.visualizer.render_online(interactive=not self.headless)
                if self.cfg['env'].get('record'):
                    self.visualizer.pl.screenshot(f'{self.frame_dir}/{i+1:06d}.png')
            if self.cfg['env'].get('record'):
                self.end_recording()

        elif self.cfg['env'].get('record_scenepic'):
            racket_params_reorder = [[], []]
            ball_params_reorder = [[], []]
            for i in range(nframes):
                for j in range(2):
                    racket_params_reorder[j].append(racket_params[i][j])
                    ball_params_reorder[j].append(ball_params[i]['pos'].cpu().numpy())
            
            init_args = {
                'smpl_verts': smpl_verts.cpu(),
                'racket_params': racket_params_reorder,
                'ball_params': ball_params_reorder,
            }
            os.makedirs('out/html', exist_ok=True)
            self.html_visualizer.save_animation_as_html(init_args, 
                html_path=self.cfg['env'].get('rec_fname', f"out/html/{self.cfg['args'].cfg}_{eg_id:03}.html"))