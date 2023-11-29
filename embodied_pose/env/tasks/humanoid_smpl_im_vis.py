# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from mujoco_py import load_model_from_path, MjSim
import glfw
import imageio
from datetime import datetime

from isaacgym.torch_utils import *

from env.tasks.humanoid_smpl_im import HumanoidSMPLIM
from utils.torch_transform import ypr_euler_from_quat, angle_axis_to_quaternion
from utils.tools import create_vis_model_xml
from utils.mjviewer import MjViewer


class HumanoidSMPLIMVis(HumanoidSMPLIM):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.show_mj_viewer = not headless
        show_gym_viewer = cfg['env'].get('show_gym_viewer', False)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless or not show_gym_viewer)
        self.headless = headless
        self.cam_inited = False
        self.max_episode_length = 3000
        self.vis_mode = self.args.vis_mode
        if self.cfg['env'].get('record', False):
            if self.args.rec_once:
                self.cfg['env']['num_rec_frames'] = self._motion_lib._motion_num_frames.max()
                print(f"Recording only {self.cfg['env']['num_rec_frames']} frames")
            self.recording = True
            self.start_recording()

    def start_recording(self):
        filename = self.cfg['env'].get('rec_fname', self._video_path % datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        self.writer = imageio.get_writer(filename, fps=30, quality=8, macro_block_size=None)
        self.frame_index = 0
        print(f"============ Writing video to {filename} ============")

    def end_recording(self):
        self.writer.close()
        print(f"============ Video finished writing ============")

    def key_callback(self, key, action, mods):
        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_R:
            self.recording = not self.recording
            if self.recording:
                self.start_recording()
            else:
                self.end_recording()
        if key == glfw.KEY_C:
            print(f'cam_distance: {self.mj_viewer.cam.distance:.3f}')
            print(f'cam_elevation: {self.mj_viewer.cam.elevation:.3f}')
            print(f'cam_azimuth: {self.mj_viewer.cam.azimuth:.3f}')
        else:
            return False

        return True

    def _setup_humanoid_misc(self, humanoid_files):
        if self.show_mj_viewer:
            num_vis_humanoids = 3
            model_file = humanoid_files[0]
            # print(f"Loading model file: {model_file}")
            vis_model_file = model_file[:-4] + '_vis.xml'
            self.num_vis_capsules = 100
            self.num_vis_spheres = 100
            create_vis_model_xml(model_file, vis_model_file, num_vis_humanoids, num_vis_capsules=self.num_vis_capsules, num_vis_spheres=self.num_vis_spheres)
            self.mj_model = load_model_from_path(vis_model_file)
            self.mj_sim = MjSim(self.mj_model)
            self.mj_viewer = MjViewer(self.mj_sim)
            self.mj_data = self.mj_sim.data
            self.nq = self.mj_model.nq // num_vis_humanoids
            self.mj_viewer.render()
            self.mj_viewer.custom_key_callback = self.key_callback
            self.mj_viewer._hide_overlay = True
            glfw.restore_window(self.mj_viewer.window)
            glfw.set_window_size(self.mj_viewer.window, 1620, 1080)
        else:
            self.mj_viewer = None

    def render_vis(self, init=False):
        if self.headless: return
        self._sync_ref_motion(init)
        self.mj_viewer_setup(init)
        for _ in range(30 if self.recording else 10):
            self.mj_viewer.render()

        if self.recording:
            self.frame_index += 1
            self.writer.append_data(self.mj_viewer._read_pixels_as_in_window())
            if self.frame_index >= self.cfg['env'].get('num_rec_frames', 300):
                self.recording = False
                self.end_recording()
                quit()
    
    def mj_viewer_setup(self, init):
        self.mj_viewer.cam.lookat[:2] = self.mj_data.qpos[:2]
        self.mj_viewer.cam.lookat[0] += 0.5
        self.mj_viewer.cam.lookat[2] = 0.8

        if not self.cam_inited:
            self.mj_viewer.cam.distance = self.mj_model.stat.extent * 1.1
            self.mj_viewer.cam.azimuth = 45
            self.mj_viewer.cam.elevation = -10
            self.cam_inited = True

    def set_mj_actor_qpos(self, actor_qpos, root_pos, root_rot, dof_pos):
        actor_qpos[:3] = root_pos
        actor_qpos[3:7] = root_rot[[3, 0, 1, 2]]
        actor_qpos[7:] = dof_pos
        return

    def convert_dof_pos_to_dof_euler(self, dof_pos):
        dof_quat = angle_axis_to_quaternion(dof_pos.view(-1, 3))
        dof_euler = ypr_euler_from_quat(dof_quat)[..., [2, 1, 0]].reshape(-1)
        return dof_euler

    def _sync_ref_motion(self, init=False):
        dof_euler = self.convert_dof_pos_to_dof_euler(self._dof_pos[0])
        self.set_mj_actor_qpos(actor_qpos=self.mj_data.qpos[:self.nq], 
                               root_pos=self._rigid_body_pos[0, 0].cpu().numpy(), 
                               root_rot=self._rigid_body_rot[0, 0].cpu().numpy(), 
                               dof_pos=dof_euler.cpu().numpy())

        if not init:
            target_dof_euler = self.convert_dof_pos_to_dof_euler(self._prev_target_dof_pos[0])
            self.set_mj_actor_qpos(actor_qpos=self.mj_data.qpos[self.nq: 2 * self.nq], 
                                root_pos=self._prev_target_root_pos[0].cpu().numpy(), 
                                root_rot=self._prev_target_root_rot[0].cpu().numpy(), 
                                dof_pos=target_dof_euler.cpu().numpy())

            self.mj_data.qpos[self.nq * 2:] = self.mj_data.qpos[:self.nq]
            self.mj_data.qpos[self.nq * 2 + 2] += 1000.0
        else:
            self.mj_data.qpos[self.nq: 2 * self.nq] = self.mj_data.qpos[:self.nq]
            self.mj_data.qpos[self.nq * 2:] = self.mj_data.qpos[:self.nq]
            self.mj_data.qpos[self.nq * 2 + 2] += 1000.0

        self.mj_data.qpos[self.nq] += 1.0

        self.mj_sim.forward()
        return