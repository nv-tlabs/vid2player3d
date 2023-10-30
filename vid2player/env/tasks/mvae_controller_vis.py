from env.tasks.physics_mvae_controller import PhysicsMVAEController
from utils.racket import infer_racket_from_smpl
from utils.common import AverageMeterUnSync, concat_torch
from utils.torch_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

from smpl_visualizer.vis_sport import SportVisualizer
from smpl_visualizer.vis_scenepic import SportVisualizerHTML
from smpl_visualizer.vis import images_to_video

import os
import torch
import tempfile
import shutil
import torch.nn.functional as F
from tqdm import tqdm


class MVAEControllerVis(PhysicsMVAEController):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params, 
                         physics_engine=physics_engine, 
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.headless = headless
        self._righthand = self.cfg_v2p.get('righthand', True)

        if self.cfg['env'].get('record_scenepic'):
            self.html_visualizer = SportVisualizerHTML(self._smpl, show_ball=True, show_ball_target=True)
        elif self.cfg['env'].get('record') or not self.headless:
            init_args = {
                'num_actors': 1, 
                'sport': 'tennis',
                'camera': self.cfg['env'].get('camera', 'front'),
            }
            self.visualizer = SportVisualizer(verbose=False, gender='male',
                show_smpl=True, show_skeleton=True, show_racket=True,
                show_ball=True, show_ball_target=True,
                enable_shadow=True)
            # window size has to be set as 1000x1000 when enableing shadow due to bug in vtk
            self.visualizer.show_animation_online(
                init_args=init_args,
                window_size=(1000, 1000),
                off_screen=self.headless,
            )

        self.frame_index = 0
        self.joint_rot_all = None
        self.root_pos_all = None
        self.ball_pos_all = None
        self.target_pos_all = None
        self.phase_all = None
        self.swing_type_all = None
        self.hit_rate = AverageMeterUnSync(dim=self.num_envs)
        self.bounce_in_pos_error = AverageMeterUnSync(dim=self.num_envs)
        self.bounce_in_rate = AverageMeterUnSync(dim=self.num_envs)
        self.fh_ratio = AverageMeterUnSync(dim=self.num_envs)
        
    def post_physics_step(self):
        super().post_physics_step()

    def _compute_reset(self):
        # reset after bounce
        self._terminate_buf[:] = self.progress_buf > self.cfg['env'].get('episodeLength', 600)
        self.reset_buf[:] = self._terminate_buf
        has_contact = self._physics_player.task._has_racket_ball_contact

        in_time = (self._tar_time >= self._tar_time_total)
        self._reset_reaction_buf = (has_contact & (self._tar_action == 0) & self._physics_player.task._has_bounce & in_time) | \
            (~has_contact & in_time)
        self._reset_recovery_buf =  (self._tar_action == 1) & \
            (has_contact | (self._ball_pos[:, 1] < self._root_pos[:, 1] - 1) | (abs(self._ball_vel[:, 1]) < 2))

        self._reset_reaction_buf[self.reset_buf > 0] = True
        self._reset_recovery_buf[self.reset_buf > 0] = False
        
        self._compute_stats()
            
    def _compute_stats(self):
        env_ids = self._reset_recovery_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            hit = self._physics_player.task._has_racket_ball_contact[env_ids]
            self.hit_rate.update(hit, env_ids)

            is_fh = self._mvae_player._swing_type_cycle[env_ids] == 1
            self.fh_ratio.update(is_fh.long(), env_ids)

            self.bounce_in_rate.update(self._est_bounce_in[env_ids], env_ids)

            env_ids_in = env_ids[self._est_bounce_in[env_ids]]
            if len(env_ids_in) > 0:
                bounce_err_in = (self._est_bounce_pos[env_ids_in] - self._target_bounce_pos[env_ids_in]).norm(dim=1)
                self.bounce_in_pos_error.update(bounce_err_in, env_ids_in)
    
    def render_vis(self, init=False):
        self.frame_index += 1
        if not self.cfg['env'].get('record') and not self.cfg['env'].get('record_scenepic'):
            if self.headless: return

            smpl_motion = self._smpl(
                global_orient=self._joint_rot[:, 0].reshape(-1, 3),
                body_pose=self._joint_rot[:, 1:].reshape(-1, 69),
                betas=self.betas,
                root_trans = self._root_pos.reshape(-1, 3),
                return_full_pose=True,
                orig_joints=True
            )
        
            smpl_verts = smpl_motion.vertices.reshape(self.num_envs, -1, 3)
            joint_pos = smpl_motion.joints.reshape(self.num_envs, 24, 3)

            joint_pos_rel = joint_pos - self._root_pos.view(-1, 1, 3)
            racket_params = []
            for i in range(self.num_envs):
                racket_params += [infer_racket_from_smpl(
                    joint_pos_rel[i].cpu().numpy(), 
                    self._joint_rot[i].cpu().numpy(), 
                    sport='tennis', grip=self.cfg_v2p.get('grip'), righthand=self._righthand)]
                racket_params[-1]['root'] = self._root_pos[i].cpu().numpy()
            
            ball_params = []
            for i in range(self.num_envs):
                ball_params += [{
                    'pos': self._ball_pos[i],
                    'ang_vel': F.normalize(self._physics_player.task._ball_root_states[i, 10:13], dim=0),
                }]
            
            self.visualizer.update_scene_online(
                smpl_verts=smpl_verts, 
                joint_pos=joint_pos, 
                racket_params=racket_params,
                ball_params=ball_params,
                ball_targets=self._target_bounce_pos,
            )
            self.visualizer.render_online(interactive=not self.headless)
            return 

        self.joint_rot_all = concat_torch(self.joint_rot_all, self._joint_rot.unsqueeze(1).cpu(), dim=1)
        self.root_pos_all = concat_torch(self.root_pos_all, self._root_pos.unsqueeze(1).cpu(), dim=1)
        self.ball_pos_all = concat_torch(self.ball_pos_all, self._ball_pos.unsqueeze(1).cpu(), dim=1)
        self.target_pos_all = concat_torch(self.target_pos_all, self._target_bounce_pos.unsqueeze(1).cpu(), dim=1)
        # for fix two hand swing
        self.phase_all = concat_torch(self.phase_all, self._mvae_player._phase_pred.unsqueeze(1).cpu(), dim=1)
        self.swing_type_all = concat_torch(self.swing_type_all, self._mvae_player._swing_type_cycle.unsqueeze(1).cpu(), dim=1)

        if self.frame_index >= self.cfg['env'].get('num_rec_frames', 300):
            if self.cfg['env'].get('select_best'):
                # compute total distance traveled
                dist = (self.root_pos_all[:, 1:] - self.root_pos_all[:, :-1]).norm(dim=-1).sum(dim=-1)
                candidates = (self.bounce_in_rate.avg > 0.95) & (self.fh_ratio.avg < 0.6)
                candidate_env_ids = candidates.nonzero(as_tuple=False).flatten()
                candidate_env_ids = candidate_env_ids[torch.argsort(dist[candidates], descending=True)]
            else:
                candidate_env_ids = torch.arange(self.num_envs)

            # save results        
            num_eg = self.cfg['env'].get('num_eg', 1)
            start_eg_id = self.cfg['env'].get('start_eg_id', 1)
            for i in range(num_eg):
                env_id = candidate_env_ids[i]

                # print("Best stats from env:", env_id)
                # print("Hit rate: ", self.hit_rate.avg[env_id])
                # print("Bounce in rate: ", self.bounce_in_rate.avg[env_id])
                # print("Bounce in pos error avg: ", self.bounce_in_pos_error.avg[env_id])

                self.render_one_result(env_id, start_eg_id+i)
            exit()
            
    def render_one_result(self, env_id, eg_id):
        if self.cfg['env'].get('record'):
            self.start_recording(eg_id)

        joint_rot = self.joint_rot_all[env_id].to(self.device)
        root_pos = self.root_pos_all[env_id].to(self.device)
        ball_pos = self.ball_pos_all[env_id].to(self.device)
        target_pos = self.target_pos_all[env_id].to(self.device)

        if self.cfg_v2p.get('fix_two_hand_backhand_post'):
            phase = self.phase_all[env_id]
            swing_type = self.swing_type_all[env_id]

            need_fix = (swing_type == 2) & (phase > 2.0) & (phase < 5.0)
            joint_rotmat = angle_axis_to_rotation_matrix(joint_rot[need_fix])

            joint_rotmat = self._physics_player.task.optimize_two_hand_backhand(
                joint_rotmat, single=True, righthand=self._righthand)
            joint_rot[need_fix] = rotation_matrix_to_angle_axis(joint_rotmat)

        nframes = root_pos.shape[0]
        smpl_motion = self._smpl(
            global_orient=joint_rot[:, 0].reshape(-1, 3),
            body_pose=joint_rot[:, 1:].reshape(-1, 69),
            betas=self.betas[:1].repeat(nframes, 1).to(self.device),
            root_trans = root_pos.reshape(-1, 3),
            return_full_pose=True,
            orig_joints=True
        )
        smpl_verts = smpl_motion.vertices.reshape(nframes, -1, 3)
        joint_pos = smpl_motion.joints.reshape(nframes, 24, 3)
        joint_pos_rel = joint_pos - root_pos.view(-1, 1, 3)

        racket_params = []
        for i in range(nframes):
            racket_params += [infer_racket_from_smpl(
                joint_pos_rel[i].cpu().numpy(), 
                joint_rot[i].cpu().numpy(), 
                sport='tennis', grip=self.cfg_v2p.get('grip'), righthand=self._righthand)]
            racket_params[-1]['root'] = root_pos[i].cpu().numpy()
        
        ball_params = []
        for i in range(nframes):
            ball_params += [{'pos': ball_pos[i]}]

        print(f"Rendering motion from env {env_id}...")
        if self.cfg['env'].get('record') or not self.headless:
            for i in tqdm(range(nframes)):
                self.visualizer.update_scene_online(
                    smpl_verts=smpl_verts[i:i+1], 
                    joint_pos=joint_pos[i:i+1], 
                    racket_params=racket_params[i:i+1],
                    ball_params=ball_params[i:i+1],
                    ball_targets=target_pos[i:i+1],
                )
                self.visualizer.render_online(interactive=not self.headless)
                if self.cfg['env'].get('record'):
                    self.visualizer.pl.screenshot(f'{self.frame_dir}/{i+1:06d}.png')
            if self.cfg['env'].get('record'):
                self.end_recording()

        elif self.cfg['env'].get('record_scenepic'):
            init_args = {
                'smpl_verts': smpl_verts.cpu().unsqueeze(0),
                'racket_params': [racket_params],
                'ball_params': [ball_pos.cpu().numpy()],
                'ball_targets': [target_pos.cpu().numpy()]
            }
            os.makedirs('out/html', exist_ok=True)
            self.html_visualizer.save_animation_as_html(init_args, 
                html_path=self.cfg['env'].get('rec_fname', f"out/html/{self.cfg['args'].cfg}_{eg_id:03}.html"))
    
    def start_recording(self, eg_id=1):
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