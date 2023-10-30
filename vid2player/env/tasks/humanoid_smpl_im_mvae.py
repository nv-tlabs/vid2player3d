# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from env.tasks.humanoid_smpl import HumanoidSMPL
from utils.konia_transform import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, rotation_matrix_to_quaternion, quaternion_to_angle_axis, quaternion_to_rotation_matrix
from utils.torch_transform import heading_to_vec, rotation_matrix_to_quaternion
from utils import torch_utils
from utils.hybrik import batch_rigid_transform
from utils.pose import SMPLPose
from utils.racket import Racket
from utils.tennis_ball import *

import torch
import torch.nn.functional as F
import numpy as np
import os
import pdb
import math
from tqdm import tqdm


class HumanoidSMPLIMMVAE(HumanoidSMPL):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.cfg_v2p = cfg['env']['vid2player']
        self._is_train = self.cfg['env']['is_train']
        self.device = "cpu"
        self.args = args = cfg['args']
        if device_type == "cuda" or device_type == "GPU":
            self.device = "cuda" + ":" + str(device_id)

        self.has_shape_obs = cfg["env"].get("has_shape_obs", False)
        self.has_self_collision = cfg["env"].get("has_self_collision", False)
        self.has_racket_collision = cfg["env"].get("has_racket_collision", False)
        self.residual_force_scale = cfg["env"].get("residual_force_scale", 0.0)
        self.residual_torque_scale = cfg["env"].get("residual_torque_scale", self.residual_force_scale)
        self.kp_scale = cfg["env"].get("kp_scale", 1.0)
        self.kd_scale = cfg["env"].get("kd_scale", self.kp_scale)
        self.obs_type = cfg['env'].get('obs_type', 'joint_pos_and_angle')
        self.no_scale_action = cfg['env'].get('no_scale_action', False)

        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_sync_dt = control_freq_inv * sim_params.dt

        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert (self._num_amp_obs_steps >= 2)

        if ("enableHistObs" in cfg["env"]):
            self._enable_hist_obs = cfg["env"]["enableHistObs"]
        else:
            self._enable_hist_obs = False

        self._sub_rewards = None
        self._sub_rewards_names = None

        self.ground_tolerance = cfg['env'].get('ground_tolerance', 0.0)

        self._num_humanoid_bodies = 24
        self._humanoid_body_ids_lefthand = torch.cat([torch.arange(19), torch.arange(20, 25), torch.arange(19, 20)])
        self._racket_body_id = self._racket_body_id_true = 24
        self._racket_hand_body_id = 23
        self._racket_wrist_body_id = 22
        self._free_hand_body_id = 18
        self._head_body_id = 13
        self._lefthand = None
        if not self.cfg_v2p.get('dual_mode') == 'different' and self.cfg_v2p.get('righthand', True) == False:
            self._lefthand = -1
            self._racket_body_id_true = 19
            self._racket_hand_body_id = 18
            self._racket_wrist_body_id = 17
        elif self.cfg_v2p.get('dual_mode') == 'different' and False in self.cfg_v2p['righthand']:
            self._lefthand = 0 if self.cfg_v2p['righthand'][0] == False else 1
            self._racket_body_id_true = [24, 24]
            self._racket_body_id_true[self._lefthand] = 19
            self._racket_wrist_body_id = torch.zeros(cfg["env"]["numEnvs"], dtype=torch.long) + 22
            self._racket_wrist_body_id[self._lefthand::2] = 17

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=True)

        self._hack_motion_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._build_mujoco_smpl_transform()

        self._root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._racket_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._racket_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._racket_normal = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ball_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ball_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._ball_vspin = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._has_bounce = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._has_bounce_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._bounce_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._has_racket_ball_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._has_racket_ball_contact_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_target_rb_rot = self._rigid_body_rot[:, :self._num_humanoid_bodies].clone()
        self._target_root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._prev_target_root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._pd_target_dof_pos = torch.zeros_like(self._dof_pos)
        if not self._is_train:
            self._joint_rot = torch.zeros((self.num_envs, 24, 3), device=self.device, dtype=torch.float32)
        if self.cfg_v2p.get('vis_pd_target'):
            self._tar_joint_rot = torch.zeros((self.num_envs, 24, 3), device=self.device, dtype=torch.float32)
        self._racket = Racket('tennis', self.device, righthand=True)
        self._mvae_player = None
        self._smpl = None

        test_mode = self.cfg_v2p.get('test_mode', 'random') 
        if self._is_train or test_mode == 'random':
            traj_file = self.cfg_v2p['ball_traj_file']
        else:
            traj_file = self.cfg_v2p['ball_traj_file_test']
        if not self.cfg_v2p.get('dual_mode', False):
            print("Use ball traj file:", traj_file)
            self._ball_generator = TennisBallGeneratorOffline(
                traj_file=traj_file, sample_random=self._is_train or test_mode=='random',
                num_envs=self.num_envs)
        
        self.num_bounce = 0 
        self.num_bounce_in = 0 

    def _setup_tensors(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # num_env x 2 x 13
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # ball
        self._ball_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 1, :]
        self._prev_ball_root_states = torch.zeros_like(self._ball_root_states)
        self._ball_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.bodies_per_env = bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos_origin = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot_origin = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel_origin = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel_origin = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        self._update_rigid_body_state()
        self._prev_rigid_body_ang_vel = self._rigid_body_ang_vel.clone()

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)
        self._contact_forces_sum = torch.zeros_like(self._contact_forces)

        self.forces = torch.zeros((self.num_envs, self.bodies_per_env, 3), device=self.device, dtype=torch.float)
        self.torques = torch.zeros((self.num_envs, self.bodies_per_env, 3), device=self.device, dtype=torch.float)

    def _update_rigid_body_state(self):
        self._rigid_body_pos = self._rigid_body_pos_origin
        self._rigid_body_rot = self._rigid_body_rot_origin
        self._rigid_body_vel = self._rigid_body_vel_origin
        self._rigid_body_ang_vel = self._rigid_body_ang_vel_origin

        if self._lefthand == -1:
            self._rigid_body_pos = self._rigid_body_pos_origin[:, self._humanoid_body_ids_lefthand]
            self._rigid_body_rot = self._rigid_body_rot_origin[:, self._humanoid_body_ids_lefthand]
            self._rigid_body_vel = self._rigid_body_vel_origin[:, self._humanoid_body_ids_lefthand]
            self._rigid_body_ang_vel = self._rigid_body_ang_vel_origin[:, self._humanoid_body_ids_lefthand]
        elif self._lefthand in [0, 1]:
            self._rigid_body_pos[self._lefthand::2] = self._rigid_body_pos_origin[self._lefthand::2, self._humanoid_body_ids_lefthand]
            self._rigid_body_rot[self._lefthand::2] = self._rigid_body_rot_origin[self._lefthand::2, self._humanoid_body_ids_lefthand]
            self._rigid_body_vel[self._lefthand::2] = self._rigid_body_vel_origin[self._lefthand::2, self._humanoid_body_ids_lefthand]
            self._rigid_body_ang_vel[self._lefthand::2] = self._rigid_body_ang_vel_origin[self._lefthand::2, self._humanoid_body_ids_lefthand]

    def _build_termination_heights(self):
        head_term_height = self.cfg["env"]["terminationHeadHeight"]
        default_termination_height = self.cfg["env"]["terminationBodyHeight"]
        self._termination_heights = np.array([default_termination_height] * self.num_bodies)
        self._humanoid_head_id = head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0],
                                                                                 "Head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _build_mujoco_smpl_transform(self):
        smpl_joint_names = [ 
            "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
            "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax", 
            "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
            "L_Wrist", "R_Wrist", "L_Hand", "R_Hand", ]
        mujoco_joint_names = [
            'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
            'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
            'R_Elbow', 'R_Wrist', 'R_Hand'
        ]
        self._smpl_2_mujoco = [
            smpl_joint_names.index(q) for q in mujoco_joint_names
            if q in smpl_joint_names
        ]
        self._mujoco_2_smpl = [
            mujoco_joint_names.index(q) for q in smpl_joint_names
            if q in mujoco_joint_names
        ]
    
    def create_sim(self):
        super().create_sim()

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = ""

        self.humanoid_masses = []
        self.humanoid_assets = dict()
        self.humanoid_files = dict()

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._reset_ref_motion_bodies = torch.zeros((self.num_envs, self.cfg['env'].get('gender_beta_dim', 11)), device=self.device)
        # set gender as male (HARD CODE)
        self._reset_ref_motion_bodies[:, 0] = 1
        if self.cfg_v2p.get('dual_mode') == 'different':
            self._reset_ref_motion_bodies[::2, 1:11] = torch.FloatTensor(self.cfg_v2p['smpl_beta'][0])
            self._reset_ref_motion_bodies[1::2, 1:11] = torch.FloatTensor(self.cfg_v2p['smpl_beta'][1])
        else:
            self._reset_ref_motion_bodies[:, 1:11] = torch.FloatTensor(self.cfg_v2p['smpl_beta'])
    
        # load existing humanoid asset
        motion_ids = np.zeros(self.num_envs, dtype=int)
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if self.cfg_v2p.get('dual_mode') == 'different':
            for i, file in enumerate(asset_file):
                humanoid_asset = self.gym.load_asset(self.sim, asset_root, file, asset_options)
                self.humanoid_assets[i] = humanoid_asset
                self.humanoid_files[i] = os.path.join(asset_root, file)
            motion_ids[1::2] = 1
        else:
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            self.humanoid_assets[0] = humanoid_asset
            self.humanoid_files[0] = os.path.join(asset_root, asset_file)

        self.humanoid_asset = humanoid_asset
        self._setup_character_props(self.cfg["env"]["keyBodies"])

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        # load tennis ball asset
        ball_asset_options = gymapi.AssetOptions()
        asset_file = self.cfg["env"]["asset"].get("assetFileNameBall", 'tennis_ball.urdf')
        self.ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, ball_asset_options)

        self.envs = []
        self.humanoid_handles = []
        self.ball_handles = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.humanoid_assets[motion_ids[i]], self.ball_asset)
            self.envs.append(env_ptr)
        self.humanoid_masses = np.array(self.humanoid_masses)
        print("Humanoid weights:", np.array2string(self.humanoid_masses[:32], precision=2, separator=","))

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        self._setup_humanoid_misc(self.humanoid_files)

    def _setup_character_props(self, key_bodies):
        body_names = self.gym.get_asset_rigid_body_names(self.humanoid_asset)
        # remove racket 
        body_names = [b for b in body_names if b != 'Racket']
        dof_names = self.gym.get_asset_dof_names(self.humanoid_asset)
        self._dof_body_ids = []
        self._dof_offsets = []
        cur_dof_index = 0
        dof_body_names = [x[:-2] for x in dof_names]
        for i, body in enumerate(body_names):
            if body != dof_body_names[cur_dof_index]:
                continue
            self._dof_body_ids.append(i)
            self._dof_offsets.append(cur_dof_index)
            while body == dof_body_names[cur_dof_index]:
                cur_dof_index += 1
                if cur_dof_index >= len(dof_names):
                    break
        self._dof_offsets.append(len(dof_names))
        self._dof_obs_size = len(self._dof_body_ids) * 6

        self._num_actions = self._num_dof = len(dof_names)
        if self.residual_force_scale > 0:
            self._num_actions += 6

        num_obs_dict = {
            'joint_pos_and_angle': 1 + len(body_names) * (3 + 6 + 3 + 3) - 3 + len(body_names) * (
                        3 + 6) + 1 + 6 + 2 + 2 + self._num_dof * 2 + self._reset_ref_motion_bodies.shape[-1],
            'joint_pos': 1 + len(body_names) * (3 + 6 + 3 + 3) - 3 + len(body_names) * 3 + 1 + 2 + self._num_dof + self._reset_ref_motion_bodies.shape[-1]
        }

        self._num_obs = num_obs_dict[self.obs_type]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset, ball_asset):
        col_group = env_id
        col_filter = 0
        if not self.has_self_collision and not self.has_racket_collision:
            col_filter = 1
        col_filter_ball = 0

        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        humanoid_mass = np.sum(
            [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)])
        self.humanoid_masses.append(humanoid_mass)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)

        filter_ints = None
        if self.has_self_collision:
            assert self._righthand
            filter_ints = [
                0, 1, 0, 0, 0, 1, 0, 0, 0, 32, 6, 120, 0, 64, 0, 20, 0, 0,
                0, 0, 10, 0, 0, 0,    0, 0
            ]
        elif self.has_racket_collision:
            if self._righthand:
                filter_ints = [1]*24 + [0] * 2
            else:
                filter_ints = [1]*19 + [0] * 2 + [1] * 5
        
        if filter_ints is not None:
            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
        
        if len(props) > 24:
            assert(len(props) == 26)
            # Racket contains two body parts, the second part is racket head
            racket_body_id = self._racket_body_id_true
            if not isinstance(racket_body_id, int):
                racket_body_id = racket_body_id[env_id % 2]

            props[racket_body_id+1].restitution = self.cfg_v2p.get('restitution', 1.0)
            props[racket_body_id+1].compliance = 0.5
            props[racket_body_id+1].friction = self.cfg_v2p.get('racket_friction', 0.8)
        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            pd_scale = humanoid_mass / self.cfg['env'].get('default_humanoid_mass', 90.0)
            dof_prop['stiffness'] *= pd_scale * self.kp_scale
            dof_prop['damping'] *= pd_scale * self.kd_scale

            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        
        # create ball
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 1)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        ball_handle = self.gym.create_actor(env_ptr, ball_asset, start_pose, "ball", col_group, col_filter_ball)
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 1.0, 0.0))
    
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)
        props[0].restitution = self.cfg_v2p.get('restitution', 1.0)
        props[0].compliance = 0.5
        props[0].friction = self.cfg_v2p.get('ball_friction', 0.8)
        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, props)

        self.humanoid_handles.append(humanoid_handle)
        self.ball_handles.append(ball_handle)

    def _setup_humanoid_misc(self, humanoid_files):
        pass

    def reset(self, reset_humanoid_env_ids, reset_ball_env_ids):
        return self._reset_envs(reset_humanoid_env_ids, reset_ball_env_ids)

    def _reset_envs(self, reset_humanoid_env_ids, reset_ball_env_ids):
        if len(reset_humanoid_env_ids) > 0:
            self._reset_actors(reset_humanoid_env_ids)
        if len(reset_ball_env_ids) > 0:
            traj = self._reset_balls(reset_ball_env_ids, reset_humanoid_env_ids)
        
        self._reset_env_tensors(reset_humanoid_env_ids, reset_ball_env_ids)

        if not self.cfg['env']['is_train']:
            self._update_state_from_sim()

        return traj

    def _reset_actors(self, env_ids):
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot \
            = self._smpl_to_sim(self._mvae_player._root_pos[env_ids].clone(), self._mvae_player._joint_rotmat[env_ids], adjust_height=True)
        
        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rb_pos=rb_pos,
                            rb_rot=rb_rot)
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._rigid_body_pos[env_ids, :self._num_humanoid_bodies] = rb_pos
        self._rigid_body_rot[env_ids, :self._num_humanoid_bodies] = rb_rot
      
        self._rigid_body_vel[env_ids] = 0
        self._rigid_body_ang_vel[env_ids] = 0
        self._prev_rigid_body_ang_vel[env_ids] = 0

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        # save init state as prev_target
        self._prev_target_root_pos[env_ids] = root_pos.clone()
        self._prev_target_rb_rot[env_ids] = rb_rot.clone()

        self._root_pos[env_ids] = root_pos
        self._root_vel[env_ids] = root_vel

        # for rendering init frame
        self._pd_target_dof_pos[env_ids] = dof_pos
        self._target_root_pos[env_ids] = root_pos
    
    def _reset_balls(self, env_ids, humanoid_env_ids=None):
        ball_root_states = None
        traj, launch_pos, launch_vel, launch_vspin = self._ball_generator.generate(
            len(env_ids), need_init_state=True, start_pos=self._ball_pos[env_ids].cpu(), env_ids=env_ids)

        launch_ang_vel = launch_vspin.view(-1, 1) * math.pi * 2 * F.normalize(
            torch.cross(launch_vel, torch.FloatTensor([0, 0, -1]).repeat(len(env_ids), 1)), dim=1)
        
        if ball_root_states is not None:
            self._ball_root_states[env_ids] = ball_root_states
        else:
            self._ball_root_states[env_ids, 0:3] = launch_pos.to(self.device)
            self._ball_root_states[env_ids, 7:10] = launch_vel.to(self.device)
            self._ball_root_states[env_ids, 10:13] = launch_ang_vel.to(self.device)

        self._has_bounce[env_ids] = 0
        self._bounce_pos[env_ids] = 0
        self._has_racket_ball_contact[env_ids] = 0
        self._ball_pos[env_ids] = self._ball_root_states[env_ids, 0:3]
        self._ball_vel[env_ids] = self._ball_root_states[env_ids, 7:10]

        return traj
    
    def create_ball_state_for_serve(self, env_ids):
        def init_serve(start_pos, target_pos):
            t = 25 / 30
            start_pos = start_pos.clone()
            start_pos[:, 2] += 0.1

            vel = torch.zeros_like(start_pos)
            vel[:, :2] = (target_pos - start_pos)[:, :2] / t
            vel[:, 2] = ((target_pos - start_pos)[:, 2] + 0.5 * g * t**2) / t
            return start_pos, vel

        # get other hand position
        # init ball state
        start_pos = self._rigid_body_pos[env_ids, self._free_hand_body_id].cpu()
        target_pos = torch.FloatTensor([[-0.87, -12.10, 2.71]])
        ball_pos, ball_vel = init_serve(start_pos, target_pos)

        root_pos = ball_pos
        root_vel = ball_vel
        root_ang_vel = torch.zeros(len(env_ids), 3)
        # compute trajectory online for future ball obs
        traj = torch.zeros((len(env_ids), 100, 3), dtype=torch.float32)
        for i in range(100):
            t = i / 30
            t0 = 1 / 30
            traj[:, i, :2] = root_pos[:, :2] + root_vel[:, :2] * t
            if i == 0:
                traj[:, i, 2] = root_pos[:, 2]
                vel_y = root_vel[:, 2]
            else:
                vel_y_prev = vel_y
                vel_y = vel_y - g*t0 - kf/m * BASE_CD * vel_y ** 2 * t0
                traj[:, i, 2] = traj[:, i-1, 2] + 0.5 * (vel_y_prev + vel_y) * t0 
        
        return traj, root_pos, root_vel, root_ang_vel

    def _reset_env_tensors(self, reset_humanoid_env_ids, reset_ball_env_ids):
        if len(reset_humanoid_env_ids) > 0:
            env_ids_int32 = torch.cat([
                self._humanoid_actor_ids[reset_humanoid_env_ids], 
                self._ball_actor_ids[reset_ball_env_ids]])
        else:
            env_ids_int32 = self._ball_actor_ids[reset_ball_env_ids]
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        if len(reset_humanoid_env_ids) > 0:
            env_ids_int32 = self._humanoid_actor_ids[reset_humanoid_env_ids]
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self._dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
            self.progress_buf[reset_humanoid_env_ids] = 0
            self.reset_buf[reset_humanoid_env_ids] = 0
            self._terminate_buf[reset_humanoid_env_ids] = 0
        
    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._update_rigid_body_state()
    
    def post_mvae_step(self):  
        self._set_target_motion_state()
        self._compute_observations()
        if torch.isnan(self.obs_buf).any():
            print("Found NAN in physics player obersavations")
            pdb.set_trace()

    def _set_target_motion_state(self, env_ids=None):
        target_root_pos = self._mvae_player._root_pos.clone()
        if self.cfg_v2p.get('add_residual_root'):
            target_root_pos += self._controller._res_root_actions

        if self.cfg_v2p.get('fix_head_orientation'):
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot \
                = self._smpl_to_sim(
                    target_root_pos, self._mvae_player._joint_rotmat, 
                )
            head_rotmat = quaternion_to_rotation_matrix(rb_rot[:, self._head_body_id, [3, 0, 1, 2]])
            lookat = torch.matmul(head_rotmat, torch.FloatTensor([0, 0, 1]).to(self.device))
            lookat = F.normalize(lookat[:, :2], dim=-1)
            head_ball = F.normalize(self._ball_pos[:, :2] - rb_pos[:, self._head_body_id, :2], dim=-1)
            lookat_angle = torch.atan2(lookat[:, 1], lookat[:, 0])
            ball_angle = torch.atan2(head_ball[:, 1], head_ball[:, 0])
            angle_diff = ball_angle - lookat_angle
            angle_diff[angle_diff > pi] -= pi * 2
            angle_diff[angle_diff < -pi] += pi * 2
            # no need to correct head if miss
            miss = (self._ball_pos[:, 1] < self._root_pos[:, 1] - 0.5) | (abs(self._ball_pos[:, 0]) > 4)
            angle_diff[miss] = 0

            joint_rot = rotation_matrix_to_angle_axis(self._mvae_player._joint_rotmat[:, [SMPLPose.Head, SMPLPose.Neck]])
            joint_rot[:, 0, 1] += angle_diff / 2
            joint_rot[:, 1, 1] += angle_diff / 2

            need_fix = torch.ones(self.num_envs, dtype=torch.bool)
            # not fix head during serve
            # is_serve = self._mvae_player._swing_type_cycle == 0
            # need_fix[is_serve] = False

            new_joint_rotmat = self._mvae_player._joint_rotmat[need_fix].clone()
            new_joint_rotmat[:, [SMPLPose.Head, SMPLPose.Neck]] = angle_axis_to_rotation_matrix(joint_rot[need_fix])
            self._mvae_player._joint_rotmat[need_fix] = new_joint_rotmat
        
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot \
            = self._smpl_to_sim(
                target_root_pos, self._mvae_player._joint_rotmat, 
                prev_root_pos=self._prev_target_root_pos,
                prev_rb_rot=self._prev_target_rb_rot,
            )

        # new target
        if env_ids is None:
            self._target_root_pos = root_pos
            self._target_root_rot = root_rot
            self._target_dof_pos = dof_pos
            self._target_root_vel = root_vel
            self._target_root_ang_vel = root_ang_vel
            self._target_dof_vel = dof_vel
            self._target_rb_pos = rb_pos
            self._target_rb_rot = rb_rot
        else:
            self._target_root_pos[env_ids] = root_pos[env_ids]
            self._target_root_rot[env_ids] = root_rot[env_ids]
            self._target_dof_pos[env_ids] = dof_pos[env_ids]
            self._target_root_vel[env_ids] = root_vel[env_ids]
            self._target_root_ang_vel[env_ids] = root_ang_vel[env_ids]
            self._target_dof_vel[env_ids] = dof_vel[env_ids]
            self._target_rb_pos[env_ids] = rb_pos[env_ids]
            self._target_rb_rot[env_ids] = rb_rot[env_ids]
        
    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        dof_actions = self.actions[:, :self._num_dof]

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(dof_actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = dof_actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        # residual forces
        if self.residual_force_scale > 0:
            res_force = self.actions[:, self._num_dof: self._num_dof + 3].clone() * self.residual_force_scale
            res_torque = self.actions[:, self._num_dof + 3: self._num_dof + 6].clone() * self.residual_torque_scale
            root_rot = remove_base_rot(self._rigid_body_rot[:, 0, :])
            root_heading_q = torch_utils.calc_heading_quat(root_rot)
            self.res_force = torch_utils.my_quat_rotate(root_heading_q, res_force)
            self.res_torque = torch_utils.my_quat_rotate(root_heading_q, res_torque)
            self.forces[:, 0, :] = self.res_force
            self.torques[:, 0, :] = self.res_torque
            
        self._save_prev_target_motion_state()
        self._has_bounce_now[:] = 0
        self._has_racket_ball_contact_now[:] = 0
        self._contact_forces_sum[:] = 0
        self._prev_rigid_body_ang_vel = self._rigid_body_ang_vel.clone()

    def _action_to_pd_targets(self, action):
        pd_target_base = self.cfg['env'].get('pd_target_base', 'target_pos')
        base_dict = {
            'target_pos': self._target_dof_pos,
            'current_pos': self._dof_pos,
            'none': self._pd_action_offset
        }
        if self.no_scale_action:
            pd_tar = base_dict[pd_target_base] + action
        else:
            pd_tar = base_dict[pd_target_base] + self._pd_action_scale * action
        
        pd_lower = self._dof_pos - np.pi / 2
        pd_upper = self._dof_pos + np.pi / 2
        pd_tar = torch.maximum(torch.minimum(pd_tar, pd_upper), pd_lower)
        self._pd_target_dof_pos = pd_tar.clone()
        return pd_tar
    
    def apply_external_force_to_ball(self, ball_root_states):
        pos = ball_root_states[:, 0:3]
        vel = ball_root_states[:, 7:10]
        vel_scalar = vel.norm(dim=1).view(-1, 1)
        vel_scalar[vel_scalar == 0] += 1 # Avoid divide by 0
        vel_norm = vel / vel_scalar
        g_tensor = torch.FloatTensor([[0, 0, -1]]).repeat(self.num_envs, 1).to(self.device)
        vel_tan = torch.cross(vel_norm, g_tensor)
        vspin = ball_root_states[:, 10:13].norm(dim=1).view(-1, 1) / (math.pi * 2)
        spin_scale = self.cfg_v2p.get('spin_scale', 1.0)

        cd = get_cd(vel_scalar, vspin * spin_scale)
        cl = get_cl(vel_scalar, vspin * spin_scale)
        cl = cl * torch.where(vspin > 0, 
            -1 * torch.ones_like(cl),
            1 * torch.ones_like(cl)
        )
        force_drag = - kf * cd * vel_scalar * vel
        force_lift = - kf * cl * vel_scalar ** 2 * torch.cross(vel_tan, vel_norm) 
        
        if self.cfg['sim'].get('substeps', 2) > 2:
            has_bounce_now = ~self._has_bounce & (pos[:, 2] <= R * 6)
        else:
            has_bounce_now = ~self._has_bounce & (pos[:, 2] <= R * 4)
        self._has_bounce_now |= has_bounce_now
        self._has_bounce |= has_bounce_now
        self._bounce_pos[has_bounce_now] = pos[has_bounce_now]
        ball_forces = force_drag + force_lift
        self.forces[:, -1, :] = ball_forces 
    
    def _save_prev_target_motion_state(self):
        # previous target
        self._prev_target_root_pos = self._target_root_pos.clone()
        self._prev_target_root_rot = self._target_root_rot.clone()
        self._prev_target_dof_pos = self._target_dof_pos.clone()
        self._prev_target_root_vel = self._target_root_vel.clone()
        self._prev_target_root_ang_vel = self._target_root_ang_vel.clone()
        self._prev_target_dof_vel = self._target_dof_vel.clone()
        self._prev_target_rb_pos = self._target_rb_pos.clone()
        self._prev_target_rb_rot = self._target_rb_rot.clone()
    
    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.apply_external_force_to_ball(self._ball_root_states)

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), 
                gymtorch.unwrap_tensor(self.torques), gymapi.ENV_SPACE)

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            if not self._is_train:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self._update_rigid_body_state()
                self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            self.forces[:] = 0
            self.torques[:] = 0

            if self.cfg['sim'].get('substeps', 2) <= 2:
                assert self._lefthand not in [0, 1]
                
                # contact force sensor are good when using small substeps
                has_racket_ball_contact_now = ~self._has_racket_ball_contact & \
                    (self._contact_forces[:, self._racket_body_id_true].sum(dim=-1) != 0) & \
                    (self._contact_forces[:, -1].sum(dim=-1) != 0)
                self._has_racket_ball_contact_now |= has_racket_ball_contact_now
                self._has_racket_ball_contact |= has_racket_ball_contact_now

            self._contact_forces_sum += self._contact_forces

            self.render()
        
    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._update_state_from_sim()

        self.extras["terminate"] = self._terminate_buf
        self.extras["sub_rewards"] = self._sub_rewards
        self.extras["sub_rewards_names"] = self._sub_rewards_names

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

    def _update_state_from_sim(self):
        if self.cfg['sim'].get('substeps', 2) > 2:
            # contact force sensor are bad when using large substeps
            # instead detect whether contact happens given ball velocity direction change
            # this may also detect contact with body or racket handle
            has_racket_ball_contact_now = ~self._has_racket_ball_contact & \
                (self._ball_root_states[:, 8] > 0) & \
                ((self._ball_root_states[:, 8] - self._ball_vel[:, 1]) > 10)
            self._has_racket_ball_contact_now = has_racket_ball_contact_now
            self._has_racket_ball_contact |= has_racket_ball_contact_now
        
        self._root_pos[:] = self._rigid_body_pos[:, 0]
        self._root_vel[:] = self._humanoid_root_states[:, 7:10]

        if not self._is_train:
            root_rot_quat=self._rigid_body_rot[:, 0][..., [3, 0, 1, 2]]
            root_rot = quaternion_to_angle_axis(root_rot_quat)
            joint_rot = torch.cat((
                root_rot.reshape(self.num_envs, 1, 3), 
                self._dof_pos.reshape(self.num_envs, -1, 3)), dim=1
                )[:, self._mujoco_2_smpl]
            self._joint_rot[:] = joint_rot
        
        if self.cfg_v2p.get('vis_pd_target'):
            self._tar_joint_rot = torch.cat((
                root_rot.reshape(self.num_envs, 1, 3), 
                self._pd_target_dof_pos.reshape(self.num_envs, -1, 3)), dim=1
                )[:, self._mujoco_2_smpl]
        
        self._racket_pos[:] = self._rigid_body_pos[:, self._racket_body_id]
        self._racket_vel[:] = self._rigid_body_vel[:, self._racket_body_id]
        if self._lefthand not in [0, 1]:
            rb_rotmat_wrist = quaternion_to_rotation_matrix(self._rigid_body_rot[:, self._racket_wrist_body_id][..., [3, 0, 1, 2]])
        else:
            rb_rotmat_wrist = quaternion_to_rotation_matrix(
                self._rigid_body_rot[torch.arange(self.num_envs), self._racket_wrist_body_id][..., [3, 0, 1, 2]])
        racket_normal_dict = {
            'eastern': [0, 1, 0],
            'semi_western': [0, 1./math.sqrt(2), 1./math.sqrt(2)],
        }
        if self.cfg_v2p.get('dual_mode') == 'different':
            grip = self.cfg_v2p['grip']
            normal = torch.FloatTensor([racket_normal_dict[grip[0]], racket_normal_dict[grip[1]]]).repeat(self.num_envs//2, 1).unsqueeze(-1)
            self._racket_normal[:] = torch.matmul(rb_rotmat_wrist, normal.to(self.device))[..., 0]
        else:
            normal = racket_normal_dict[self.cfg_v2p.get('grip', 'eastern')]
            self._racket_normal[:] = torch.matmul(rb_rotmat_wrist, torch.FloatTensor(normal).to(self.device))

        self._ball_pos[:] = self._ball_root_states[:, 0:3]
        self._ball_vel[:] = self._ball_root_states[:, 7:10]
        self._ball_vspin[:] = self._ball_root_states[:, 10:13].norm(dim=1) / (math.pi * 2)

        if self.cfg_v2p.get('update_mvae_from_sim', False):
            # update mvae root state/condition with sim state 
            self._mvae_player._root_pos[:] = self._rigid_body_pos[:, 0]
            self._mvae_player._root_vel[:] = self._humanoid_root_states[:, 7:10] * self.dt

            assert self._mvae_player._conditions.shape[1] == 1
            self._mvae_player._conditions[:, :, self._mvae_player._root_pos_inds] = self._mvae_player._mvae.normalize(
                self._mvae_player._root_pos, self._mvae_player._root_pos_inds).view(self.num_envs, 1, 3)
            self._mvae_player._conditions[:, :, self._mvae_player._root_vel_inds] = self._mvae_player._mvae.normalize(
                self._mvae_player._root_vel, self._mvae_player._root_vel_inds).view(self.num_envs, 1, 3)
        
    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos[:, :self._num_humanoid_bodies]
            body_rot = self._rigid_body_rot[:, :self._num_humanoid_bodies]
            body_vel = self._rigid_body_vel[:, :self._num_humanoid_bodies]
            body_ang_vel = self._rigid_body_ang_vel[:, :self._num_humanoid_bodies]
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            target_pos = self._target_rb_pos
            target_rot = self._target_rb_rot
            target_dof_pos = self._target_dof_pos
            motion_bodies = self._reset_ref_motion_bodies
        else:
            body_pos = self._rigid_body_pos[env_ids, :self._num_humanoid_bodies]
            body_rot = self._rigid_body_rot[env_ids, :self._num_humanoid_bodies]
            body_vel = self._rigid_body_vel[env_ids, :self._num_humanoid_bodies]
            body_ang_vel = self._rigid_body_ang_vel[env_ids, :self._num_humanoid_bodies]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            target_pos = self._target_rb_pos[env_ids]
            target_rot = self._target_rb_rot[env_ids]
            target_dof_pos = self._target_dof_pos[env_ids]
            motion_bodies = self._reset_ref_motion_bodies[env_ids]

        obs_func_dict = {
            'joint_pos_and_angle': compute_humanoid_observations_imitation,
            'joint_pos': compute_humanoid_observations_imitation_jpos
        }
        obs_func = obs_func_dict[self.obs_type]

        obs = obs_func(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel,
                       target_dof_pos, body_vel, body_ang_vel, motion_bodies, self._local_root_obs,
                       self._root_height_obs)
        return obs

    def _smpl_to_sim(self, root_pos, joint_rotmat, prev_root_pos=None, prev_rb_rot=None,
        adjust_height=True, ground_tolerance=0):
        batch_size = root_pos.shape[0]
        dof_pos = rotation_matrix_to_angle_axis(joint_rotmat)
        dof_pos = dof_pos[:, self._smpl_2_mujoco, :][:, 1:, :].reshape(-1, 23 * 3)

        rb_pos, rb_rot = self._forward_kinematics(
            joint_rotmat, self._smpl.joint_pos_bind[:batch_size], self._smpl.parents)
        root_rot = rb_rot[:, 0, :]

        # add back smpl trans
        root_diff = self._smpl.joint_pos_bind[:batch_size, :1] - root_pos.unsqueeze(1)
        rb_pos = rb_pos - root_diff

        if prev_root_pos is not None:
            # compute root velocity
            root_vel = (root_pos - prev_root_pos) / self.dt
            # compute joint angular velocity
            diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(prev_rb_rot), rb_rot)
            diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
            dof_vel = diff_axis * diff_angle.unsqueeze(-1) / self.dt

            root_ang_vel = dof_vel[:, 0, :] / self.dt
            dof_vel = dof_vel[:, 1:, :].reshape(-1, 23 * 3)
        else:
            root_vel = torch.zeros_like(root_pos)
            root_ang_vel = torch.zeros_like(root_pos)
            dof_vel = torch.zeros_like(dof_pos)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot

    def _forward_kinematics(self, joint_rotmat, rest_body_pos, parents):
        """
        Forward kinematics of the SMPL model.
        Args:
            rotmat: [N, 24, 3, 3] torch.Tensor, local joint rotation matrix
            rest_body_pos: [N, 24, 3] torch.Tensor, rest global joint position.
            smpl_parents: [24] torch.Tensor, parent indices of the joints.
        Returns:
            body_pos: [N, 24, 3] torch.Tensor, global joint position.
            body_rot: [N, 24, 3] torch.Tensor, global joint rotation
        """
        rb_pos, rb_rotmat = batch_rigid_transform(
            joint_rotmat, rest_body_pos, parents)
        rb_rot = rotation_matrix_to_quaternion(rb_rotmat.contiguous())[..., [1, 2, 3, 0]]

        rb_pos = rb_pos[:, self._smpl_2_mujoco, :]
        rb_rot = rb_rot[:, self._smpl_2_mujoco, :]

        return rb_pos, rb_rot
    
    def optimize_two_hand_backhand(self, joint_rotmat, single=False, righthand=True):
        root_pos = torch.zeros((joint_rotmat.shape[0], 3)).to(self.device)
        _, _, _, _, _, _, rb_pos, _, _ = self._smpl_to_sim(root_pos, joint_rotmat)

        if righthand:
            racket_hand_body_id = 23
            racket_wrist_body_id = 22
            free_hand_body_id = 18
        else:
            racket_hand_body_id = 18
            racket_wrist_body_id = 17
            free_hand_body_id = 23

        target_pos = 2 * rb_pos[:, racket_hand_body_id] \
            - rb_pos[:, racket_wrist_body_id] - rb_pos[:, 0]

        # root_pos all 0
        if righthand:
            ik_joint_smpl = [SMPLPose.LWrist, SMPLPose.LElbow, SMPLPose.LShoulder, SMPLPose.LCollar]
        else:
            ik_joint_smpl = [SMPLPose.RWrist, SMPLPose.RElbow, SMPLPose.RShoulder, SMPLPose.RCollar]
        joint_rot_ik = rotation_matrix_to_angle_axis(joint_rotmat[:, ik_joint_smpl])
        joint_rot_ik_origin = joint_rot_ik.clone()

        device = torch.device('cuda:0')
        if not single:
            lr = 0.05
            niters = 50
            loss_weight = {
                'target': 1,
                'reg': 0.1,
            }
        else:
            lr = 0.005
            niters = 500
            loss_weight = {
                'target': 1,
                'reg': 0.2,
                'smooth': 0.5,
            }
        
        # init variable
        rot_delta = torch.zeros_like(joint_rot_ik, dtype=torch.float32, device=device)
        rot_delta.requires_grad_(True)
        param_list = [rot_delta]

        # init optimizer
        optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        L1loss = torch.nn.L1Loss()

        # run optimization for N steps
        loss_dict = None
        def closure():
            nonlocal loss_dict
            loss_dict = {}
            optimizer.zero_grad()

            joint_rot_ik = joint_rot_ik_origin + rot_delta
            joint_rotmat[:, ik_joint_smpl] = angle_axis_to_rotation_matrix(joint_rot_ik)

            _, _, _, _, _, _, rb_pos, _, _ = self._smpl_to_sim(root_pos, joint_rotmat)
            current_pos = rb_pos[:, free_hand_body_id]

            loss_dict['target'] = L1loss(current_pos, target_pos.detach()) * loss_weight['target']
            
            if 'reg' in loss_weight:
                loss_dict['reg'] = L1loss(rot_delta, torch.zeros_like(rot_delta).detach()) * loss_weight['reg']

            if 'smooth' in loss_weight:
                loss_dict['smooth'] = L1loss(rot_delta[1:], rot_delta[:-1].detach()) * loss_weight['smooth']

            loss = sum(loss_dict.values())
            loss.backward(retain_graph=True)
            return loss
        
        print("Optimizing two hand swing ...")
        for iter in tqdm(range(niters)):
            optimizer.step(closure)
        print(loss_dict['target'].item(), loss_dict['reg'].item(), loss_dict['smooth'].item())

        # update final result
        joint_rot_ik = joint_rot_ik_origin + rot_delta
        joint_rotmat[:, ik_joint_smpl] = angle_axis_to_rotation_matrix(joint_rot_ik)
        return joint_rotmat.detach()
    
    def render_vis(self, init=True):
        NotImplemented 
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def remove_base_rot(quat):
    # ZL: removing the base rotation for SMPL model
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    return quat_mul(quat, base_rot.repeat(quat.shape[0], 1))

@torch.jit.script
def compute_humanoid_observations_imitation(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel,
                                            target_dof_pos, body_vel, body_ang_vel, motion_bodies, local_root_obs,
                                            root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    root_rot = remove_base_rot(root_rot)
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                  heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],
                                                 local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],
                                                 local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0],
                                                         body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0],
                                                         body_ang_vel.shape[1] * body_ang_vel.shape[2])

    """target"""
    # target root height    [N, 1]
    target_root_pos = target_pos[:, 0, :]
    target_root_rot = target_rot[:, 0, :]
    target_rel_root_h = root_h - target_root_pos[:, 2:3]
    # target root rotation  [N, 6]
    target_root_rot = remove_base_rot(target_root_rot)
    target_heading_rot, target_heading = torch_utils.calc_heading_quat_inv_with_heading(target_root_rot)
    target_rel_root_rot = quat_mul(target_root_rot, quat_conjugate(root_rot))
    target_rel_root_rot_obs = torch_utils.quat_to_tan_norm(target_rel_root_rot)
    # target 2d pos [N, 2]
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]
    # target heading    [N, 2]
    target_rel_heading = target_heading - heading
    target_rel_heading_vec = heading_to_vec(target_rel_heading)
    # target target dof   [N, dof]
    target_rel_dof_pos = target_dof_pos - dof_pos
    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
                                                           target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0],
                                                           target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])
    # target body rot   [N, 6xB]
    target_rel_body_rot = quat_mul(quat_conjugate(body_rot), target_rot)
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4)).view(
        target_rel_body_rot.shape[0], -1)

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_root_rot_obs, target_rel_2d_pos, target_rel_heading_vec,
                     target_rel_dof_pos, target_rel_body_pos, target_rel_body_rot_obs, motion_bodies), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_imitation_jpos(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel,
                                                 target_dof_pos, body_vel, body_ang_vel, motion_bodies, local_root_obs,
                                                 root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    root_rot = remove_base_rot(root_rot)
    heading_rot, heading = torch_utils.calc_heading_quat_inv_with_heading(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                  heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],
                                                 local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],
                                                 local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0],
                                                         body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0],
                                                         body_ang_vel.shape[1] * body_ang_vel.shape[2])

    """target"""
    # target root height    [N, 1]
    target_root_pos = target_pos[:, 0, :]
    target_rel_root_h = root_h - target_root_pos[:, 2:3]
    # target 2d pos [N, 2]
    target_rel_pos = target_root_pos[:, :3] - root_pos[:, :3]
    target_rel_pos = torch_utils.my_quat_rotate(heading_rot, target_rel_pos)
    target_rel_2d_pos = target_rel_pos[:, :2]
    # target body pos   [N, 3xB]
    target_rel_body_pos = target_pos - body_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
                                                           target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0],
                                                           target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_2d_pos, target_rel_body_pos, motion_bodies), dim=-1)
    return obs