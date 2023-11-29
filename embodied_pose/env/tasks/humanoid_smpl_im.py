# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import shutil
import time
import numpy as np
import os
from enum import Enum
from uuid import uuid4
from tqdm import tqdm
import glob
from scipy.stats import norm

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch
from env.tasks.humanoid_smpl import HumanoidSMPL, dof_to_obs
from utils.torch_transform import heading_to_vec, angle_axis_to_rot6d
from utils import torch_utils
from uhc.smpllib.smpl_local_robot import Robot


TMP_SMPL_DIR = f"/tmp/smpl_humanoid_{uuid4()}"


def _create_smpl_humanoid_xml(humanoid_ids, queue, smpl_robot, motion_bodies, body_scales, pid):
    res = []
    for idx in humanoid_ids:
        model_xml_path = f"{TMP_SMPL_DIR}/smpl_humanoid_{idx}.xml"
        gender_beta = motion_bodies[idx]
        rest_joints = smpl_robot.load_from_skeleton(betas=gender_beta[None, 1:], gender=gender_beta[:1], scale=body_scales[idx], model_xml_path=model_xml_path)
        smpl_robot.write_xml(model_xml_path)
        res.append((idx, model_xml_path, rest_joints))

    if not queue is None:
        queue.put(res)
    else:
        return res


class HumanoidSMPLIM(HumanoidSMPL):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self.cfg = cfg
        self.device = "cpu"
        self.model = None
        self.args = args = cfg['args']
        if device_type == "cuda" or device_type == "GPU":
            self.device = "cuda" + ":" + str(device_id)
        
        self.has_shape_obs = cfg["env"].get("has_shape_obs", False)
        self.has_self_collision = cfg["env"].get("has_self_collision", False)
        self.residual_force_scale = cfg["env"].get("residual_force_scale", 0.0)
        self.residual_torque_scale = cfg["env"].get("residual_torque_scale", self.residual_force_scale)
        self.kp_scale = cfg["env"].get("kp_scale", 1.0)
        self.kd_scale = cfg["env"].get("kd_scale", self.kp_scale)
        self.obs_type = cfg['env'].get('obs_type', 'joint_pos_and_angle')
        self.context_length = cfg['env'].get('context_length', 32)
        self.context_padding = cfg['env'].get('context_padding', 8)
        self.truncate_time = cfg['env'].get('truncate_time', True)
        self.pd_tar_lim = cfg['env'].get('pd_tar_lim', 0.5) * np.pi

        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_sync_dt = control_freq_inv * sim_params.dt
            
        if args.test:
            cfg["env"]["stateInit"] = 'Start'
            if 'test_motion_file' in cfg['env']:
                cfg['env']['motion_file'] = cfg['env']['test_motion_file']
        
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidSMPLIM.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        if ("enableHistObs" in cfg["env"]):
            self._enable_hist_obs = cfg["env"]["enableHistObs"]
        else:
            self._enable_hist_obs = False

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._sub_rewards = None
        self._sub_rewards_names = None

        self.ground_tolerance = cfg['env'].get('ground_tolerance', 0.0)
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        body_weights = cfg['env'].get('body_pos_weights', dict())
        self.body_pos_weights = torch.ones(self.num_bodies, device=self.device)
        for val, bodies in body_weights.items():
            for body in bodies:
                ind = self.body_names.index(body)
                self.body_pos_weights[ind] = val

        return

    def register_model(self, model):
        self.model = model

    def pre_epoch(self, epoch):
        return

    def pre_physics_step(self, actions):
        actions[self.reset_buf == 1] = 0

        self.actions = actions.to(self.device).clone()
        dof_actions = self.actions[:, :self._num_dof]

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(dof_actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
            self.pd_torque = (pd_tar - self._dof_pos) * self.stiffness
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
            res_force = torch_utils.my_quat_rotate(root_heading_q, res_force)
            res_torque = torch_utils.my_quat_rotate(root_heading_q, res_torque)

            forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            forces[:, 0, :] = res_force
            torques[:, 0, :] = res_torque
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        self._save_prev_target_motion_state()
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        self.body_names = body_names = self.gym.get_asset_rigid_body_names(self.humanoid_asset)
        self.dof_names = dof_names = self.gym.get_asset_dof_names(self.humanoid_asset)
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
        
        num_bodies = len(body_names)
        shape_dict = {
            'body_pos': (num_bodies, 3),
            'body_pos_gt': (num_bodies, 3),
            'body_rot': (num_bodies, 4),
            'dof_pos': (self._num_dof,),
            'dof_pos_gt': (self._num_dof,),
            'dof_vel': (self._num_dof,),
            'body_vel': (num_bodies, 3),
            'body_ang_vel': (num_bodies, 3),
            'motion_bodies': (self._motion_lib._motion_bodies.shape[-1],),
            'joint_conf': (num_bodies,)
        }

        self.obs_names = ['body_pos', 'body_rot', 'dof_pos', 'dof_vel', 'body_vel', 'body_ang_vel', 'motion_bodies']
        self.obs_shapes = [shape_dict[x] for x in self.obs_names]
        self.obs_dims = [np.prod(x) for x in self.obs_shapes]

        self.context_names = ['body_pos', 'body_rot', 'dof_pos', 'body_pos_gt', 'dof_pos_gt']
        if 'transform_specs' in self.cfg['env']:
            self.context_names.append('joint_conf')

        self.context_shapes = [shape_dict[x] for x in self.context_names]
        self.context_dims = [np.prod(x) for x in self.context_shapes]

        self.is_env_dim_setup = False

        self._num_obs = sum(self.obs_dims)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        return

    def _build_termination_heights(self):
        head_term_height = self.cfg["env"]["terminationHeadHeight"]
        default_termination_height = self.cfg["env"]["terminationBodyHeight"]
        self._termination_heights = np.array([default_termination_height] * self.num_bodies)
        self._humanoid_head_id = head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "Head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _resample_amass_motions(self):
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self._setup_tensors()

    def _create_envs(self, num_envs, spacing, num_per_row):

        robot_cfg = {
            "mesh": True,
            "model": "smpl",
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
        }

        smpl_robot = Robot(
            robot_cfg,
            data_dir= "data/smpl",
        )

        if self.cfg['env'].get('sample_first_motions', False):
            self._reset_ref_motion_ids = torch.arange(self.num_envs, device=self._motion_lib._device) % self._motion_lib.num_motions()
        else:
            weights_from_lenth = self.cfg['env'].get('motion_weights_from_length', False)
            self._reset_ref_motion_ids = self._motion_lib.sample_motions(num_envs, weights_from_lenth=weights_from_lenth)
        if 'motion_id' in self.cfg['env']:
            self._reset_ref_motion_ids[:] = self.cfg['env']['motion_id']
        self._reset_ref_motion_bodies = self._motion_lib._motion_bodies[self._reset_ref_motion_ids].to(self.device)
        all_motion_bodies = self._motion_lib._motion_bodies.cpu()
        all_motion_body_scales = self._motion_lib._motion_body_scales.cpu().numpy() if hasattr(self._motion_lib, '_motion_body_scales') else np.ones(all_motion_bodies.shape[0])
        motion_ids = self._reset_ref_motion_ids.cpu().numpy()

        print('sampled motion ids:', self._reset_ref_motion_ids)

        unique_motion_ids = np.unique(motion_ids)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = ""

        self.humanoid_masses = []
        self.humanoid_assets = dict()
        self.humanoid_files = dict()
        self.humanoid_rest_joints = dict()

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        t_start = time.time()
        res_acc = []
        for _, idx in enumerate(tqdm(unique_motion_ids)):
            res_acc += (_create_smpl_humanoid_xml([idx], None, smpl_robot, all_motion_bodies, all_motion_body_scales, 0))
        t_finish_gen = time.time()
        print(f"Finished generating {len(unique_motion_ids)} humanoids in {t_finish_gen - t_start:.3f}s!")

        for humanoid_config in res_acc:
            humanoid_idx, asset_file_real, rest_joints = humanoid_config
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file_real, asset_options)

            # create force sensors at the feet
            right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
            left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
            sensor_pose = gymapi.Transform()

            self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
            self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
            self.humanoid_assets[humanoid_idx] = humanoid_asset
            self.humanoid_files[humanoid_idx] = asset_file_real
            self.humanoid_rest_joints[humanoid_idx] = rest_joints

        self.humanoid_asset = humanoid_asset = next(iter(self.humanoid_assets.values()))
        self._setup_character_props(self.cfg["env"]["keyBodies"])

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.smpl_rest_joints = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self._build_env(i, env_ptr, self.humanoid_assets[motion_ids[i]])
            self.envs.append(env_ptr)
            self.smpl_rest_joints.append(self.humanoid_rest_joints[motion_ids[i]])
        self.smpl_rest_joints = torch.from_numpy(np.stack(self.smpl_rest_joints)).to(self.device)
        self.smpl_parents = smpl_robot.smpl_parser.parents.to(self.device)
        self.smpl_children = smpl_robot.smpl_parser.children_map.to(self.device)
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

        shutil.rmtree(TMP_SMPL_DIR, ignore_errors=True)

        print(f"Finished loading {num_envs} humanoids in {time.time() - t_finish_gen:.3f}s!")
        return

    def _setup_humanoid_misc(self, humanoid_files):
        pass

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = 0
        if not self.has_self_collision:
            col_filter = 1

        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        humanoid_mass = np.sum([prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)])
        self.humanoid_masses.append(humanoid_mass)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            pd_scale = humanoid_mass / self.cfg['env'].get('default_humanoid_mass', 90.0)
            dof_prop['stiffness'] *= pd_scale * self.kp_scale
            dof_prop['damping'] *= pd_scale * self.kd_scale
            self.stiffness = torch.from_numpy(dof_prop['stiffness']).to(self.device)
            self.damping = torch.from_numpy(dof_prop['damping']).to(self.device)

            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _action_to_pd_targets(self, action):
        pd_tar = action
        pd_lower = self._dof_pos - self.pd_tar_lim
        pd_upper = self._dof_pos + self.pd_tar_lim
        pd_tar = torch.maximum(torch.minimum(pd_tar, pd_upper), pd_lower)
        return pd_tar

    def post_physics_step(self):
        self.progress_buf += 1
        self._rigid_body_states_simulated = True

        self._cur_ref_motion_times += self.dt
        self._set_target_motion_state()

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf
        self.extras["sub_rewards"] = self._sub_rewards
        self.extras["sub_rewards_names"] = self._sub_rewards_names

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        
        return

    def _load_motion(self, motion_file):
        gpu_motion_lib = self.cfg['env'].get('gpu_motion_lib', True)
        device = self.device if gpu_motion_lib else 'cpu'
        
        if os.path.isdir(motion_file):
            self.motion_lib_files = sorted(glob.glob(f'{motion_file}/*.pth'))
            motion_file_range = self.cfg['env'].get('motion_file_range', None)
            if motion_file_range is not None:
                self.motion_lib_files = self.motion_lib_files[motion_file_range[0]:motion_file_range[1]]
            motion_libs = [torch.load(f, map_location=device) for f in self.motion_lib_files]
            self._motion_lib = motion_libs[0]
            self._motion_lib.merge_multiple_motion_libs(motion_libs[1:])
            print(f'Loading motion files to {device}:')
            for f in self.motion_lib_files:
                print(f)
        else:
            self.motion_lib_files = [motion_file]
            self._motion_lib = torch.load(motion_file, map_location=device)
            print(f'Loading motion file to {device}: {motion_file}')
        self._motion_lib._device = device
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._state_reset_happened = True

        super()._reset_envs(env_ids)

        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self._state_reset_happened:
            env_ids = self._reset_ref_env_ids
            self._rigid_body_pos[env_ids] = self._reset_rb_pos
            self._rigid_body_rot[env_ids] = self._reset_rb_rot
            self._rigid_body_vel[env_ids] = 0
            self._rigid_body_ang_vel[env_ids] = 0
            self._state_reset_happened = False

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidSMPLIM.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidSMPLIM.StateInit.Start
              or self._state_init == HumanoidSMPLIM.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidSMPLIM.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        use_env_ids = not (len(env_ids) == self.num_envs and torch.all(env_ids == torch.arange(self.num_envs, device=self.device)))
        num_envs = env_ids.shape[0]
        motion_ids = self._reset_ref_motion_ids
        if use_env_ids:
            motion_ids = motion_ids[env_ids]

        if (self._state_init == HumanoidSMPLIM.StateInit.Random
            or self._state_init == HumanoidSMPLIM.StateInit.Hybrid):
            truncate_time = self.context_length * self.dt if self.truncate_time else None
            motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time).to(self.device)
        elif (self._state_init == HumanoidSMPLIM.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times.to(self._motion_lib._device), return_rigid_body=True, device=self.device, adjust_height=True, ground_tolerance=self.ground_tolerance)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel,
                            rb_pos=rb_pos,
                            rb_rot=rb_rot)

        self._reset_ref_env_ids = env_ids
        if not use_env_ids:
            self._reset_ref_motion_times = motion_times
            self._cur_ref_motion_times = self._reset_ref_motion_times.clone()
        else:
            self._reset_ref_motion_times[env_ids] = motion_times
            self._cur_ref_motion_times[env_ids] = motion_times
        self._set_target_motion_state(env_ids=env_ids if use_env_ids else None)

        self._init_context(motion_ids, motion_times)
        return

    def _init_context(self, motion_ids, motion_times):
        motion_times = motion_times + self.dt
        context_padded_length = self.context_length + self.context_padding * 2
        all_motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, context_padded_length])
        time_steps = self.dt * torch.arange(-self.context_padding, self.context_length + self.context_padding, device=motion_times.device)
        all_motion_times = motion_times.unsqueeze(-1) + time_steps

        all_motion_ids_flat = all_motion_ids.view(-1)
        all_motion_times_flat = all_motion_times.view(-1)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot \
               = self._motion_lib.get_motion_state(all_motion_ids_flat, all_motion_times_flat.to(self._motion_lib._device), return_rigid_body=True, device=self.device, adjust_height=True, ground_tolerance=self.ground_tolerance)

        context_dict = {
            'body_pos': rb_pos,
            'body_rot': rb_rot,
            'dof_pos': dof_pos,
            'body_pos_gt': rb_pos.clone(),
            'dof_pos_gt': dof_pos.clone()
        }

        self._transform_target(context_dict)

        self.context_feat = torch.cat([context_dict[x].view(context_dict[x].shape[0], -1) for x in self.context_names], dim=-1)
        self.context_feat = self.context_feat.view(self.num_envs, -1, self.context_feat.shape[-1])

        self.context_mask = all_motion_times <= (self._motion_lib._motion_lengths[self._reset_ref_motion_ids] + 2 * self.dt).unsqueeze(-1)

        if self.model is not None:
            if not self.is_env_dim_setup:
                self.model.a2c_network.setup_env_named_dims(self.obs_names, self.obs_shapes, self.obs_dims, self.context_names, self.context_shapes, self.context_dims)
                self.is_env_dim_setup = True
            with torch.no_grad():
                self.model.a2c_network.forward_context(self.context_feat, self.context_mask)

    def _transform_target(self, context_dict):
        transform_specs = self.cfg['env'].get('transform_specs', dict())
        context_dict['joint_conf'] = joint_conf = torch.ones_like(context_dict['body_pos'][..., 0])
        for transform, specs in transform_specs.items():
            if transform == 'mask_joints':
                body_pos = context_dict['body_pos']
                joint_index = [self.body_names.index(joint) for joint in specs['joints']]
                joint_conf[..., joint_index] = 0.0
                context_dict['body_pos'] = body_pos * joint_conf.unsqueeze(-1)
            elif transform == 'noisy_joints':
                noise_std = torch.ones_like(context_dict['joint_conf']) * specs['noise_std']
                noise_mask = torch.bernoulli(torch.ones(context_dict['joint_conf'].shape) * specs['prob'])
                noise_std[noise_mask == 0.0] = 0.0
                noise = torch.randn_like(context_dict['body_pos']) * noise_std.unsqueeze(-1)
                noise_norm = noise.norm(dim=-1) / (np.sqrt(3) * specs['conf_std'])
                conf = (1 - torch.tensor(norm.cdf(noise_norm.cpu()), device=noise.device, dtype=noise.dtype)) * 2
                context_dict['body_pos'] += noise
                context_dict['joint_conf'] = conf
                # remove occluded joints
                occluded_joints = context_dict['joint_conf'] < specs['min_conf']
                context_dict['joint_conf'][occluded_joints] = 0.0
                context_dict['body_pos'][occluded_joints] = 0.0
            elif transform == 'mask_random_joints':
                drop_mask = torch.bernoulli(torch.ones(context_dict['joint_conf'].shape) * specs['prob']) == 1.0
                drop_mask[..., 0] = 0.0
                context_dict['joint_conf'][drop_mask] = 0.0
                context_dict['body_pos'][drop_mask] = 0.0
        return

    def _set_target_motion_state(self, env_ids=None):
        if env_ids is None:
            motion_ids = self._reset_ref_motion_ids
            motion_times = self._cur_ref_motion_times + self.dt     # next frame
        else:
            motion_ids = self._reset_ref_motion_ids[env_ids]
            motion_times = self._cur_ref_motion_times[env_ids] + self.dt     # next frame
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot \
               = self._motion_lib.get_motion_state(motion_ids, motion_times.to(self._motion_lib._device), return_rigid_body=True, device=self.device, adjust_height=True, ground_tolerance=self.ground_tolerance)
        # new target
        if env_ids is None:
            self._target_root_pos = root_pos
            self._target_root_rot = root_rot
            self._target_dof_pos = dof_pos
            self._target_root_vel = root_vel
            self._target_root_ang_vel = root_ang_vel
            self._target_dof_vel = dof_vel
            self._target_key_pos = key_pos
            self._target_rb_pos = rb_pos
            self._target_rb_rot = rb_rot
        else:
            self._target_root_pos[env_ids] = root_pos
            self._target_root_rot[env_ids] = root_rot
            self._target_dof_pos[env_ids] = dof_pos
            self._target_root_vel[env_ids] = root_vel
            self._target_root_ang_vel[env_ids] = root_ang_vel
            self._target_dof_vel[env_ids] = dof_vel
            self._target_key_pos[env_ids] = key_pos
            self._target_rb_pos[env_ids] = rb_pos
            self._target_rb_rot[env_ids] = rb_rot
        return

    def _save_prev_target_motion_state(self):
        # previous target
        self._prev_target_root_pos = self._target_root_pos.clone()
        self._prev_target_root_rot = self._target_root_rot.clone()
        self._prev_target_dof_pos = self._target_dof_pos.clone()
        self._prev_target_root_vel = self._target_root_vel.clone()
        self._prev_target_root_ang_vel = self._target_root_ang_vel.clone()
        self._prev_target_dof_vel = self._target_dof_vel.clone()
        self._prev_target_key_pos = self._target_key_pos.clone()
        self._prev_target_rb_pos = self._target_rb_pos.clone()
        self._prev_target_rb_rot = self._target_rb_rot.clone()

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _compute_humanoid_obs(self, env_ids=None):
        obs_dict = {
            'body_pos': self._rigid_body_pos,
            'body_rot': self._rigid_body_rot,
            'body_vel': self._rigid_body_vel,
            'body_ang_vel': self._rigid_body_ang_vel,
            'dof_pos': self._dof_pos,
            'dof_vel': self._dof_vel,
            'target_pos': self._target_rb_pos,
            'target_rot': self._target_rb_rot,
            'target_dof_pos': self._target_dof_pos,
            'motion_bodies': self._reset_ref_motion_bodies
        }
        obs = [(obs_dict[x] if env_ids is None else obs_dict[x][env_ids]) for x in self.obs_names]
        obs = torch.cat([x.reshape(x.shape[0], -1) for x in obs], dim=-1)
        return obs

    def _compute_reward(self, actions):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel
        dof_pos = self._dof_pos
        dof_vel = self._dof_vel
        target_pos = self._prev_target_rb_pos
        target_rot = self._prev_target_rb_rot
        target_dof_pos = self._prev_target_dof_pos
        target_dof_vel = self._prev_target_dof_vel

        reward_specs = {'k_dof': 60, 'k_vel': 0.2, 'k_pos': 100, 'k_rot': 40, 'w_dof': 0.6, 'w_vel': 0.1, 'w_pos': 0.2, 'w_rot': 0.1}
        cfg_reward_specs = self.cfg['env'].get('reward_specs', dict())
        reward_specs.update(cfg_reward_specs)

        self.rew_buf[:], self._sub_rewards, self._sub_rewards_names = compute_humanoid_reward(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, 
                                                                                              target_dof_pos, target_dof_vel, body_vel, body_ang_vel, self._dof_obs_size, self._dof_offsets, self.body_pos_weights, reward_specs)
        reset_mask = self.reset_buf == 1
        if torch.any(reset_mask):
            self.rew_buf[reset_mask] = 0
            self._sub_rewards[reset_mask] = 0
        return

    def get_aux_losses(self, model_res_dict):
        aux_loss_specs = self.cfg['env'].get('aux_loss_specs', dict())
        context = model_res_dict['extra']['context']
        aux_losses = {}
        aux_losses_weighted = {}

        # dof loss
        w_dof = aux_loss_specs.get('w_dof', 0.0)
        if w_dof > 0:
            dof_pos = context['dof_pos']
            target_dof_pos = context['dof_pos_gt']
            dof_obs = angle_axis_to_rot6d(dof_pos.view(*dof_pos.shape[:-1], -1, 3)).view(*dof_pos.shape[:-1], -1)
            target_dof_obs = angle_axis_to_rot6d(target_dof_pos.view(*target_dof_pos.shape[:-1], -1, 3)).view(*target_dof_pos.shape[:-1], -1)
            diff_dof_obs = dof_obs - target_dof_obs
            dof_obs_loss = (diff_dof_obs ** 2).mean()
            aux_losses['aux_dof_rot6d_loss'] = dof_obs_loss
            aux_losses_weighted['aux_dof_rot6d_loss'] = dof_obs_loss * w_dof

        # body pos loss
        w_pos = aux_loss_specs.get('w_pos', 0.0)
        if w_pos > 0:
            body_pos = context['body_pos']
            target_pos = context['body_pos_gt']
            diff_body_pos = target_pos - body_pos
            diff_body_pos = diff_body_pos * self.body_pos_weights[:, None]
            body_pos_loss = (diff_body_pos ** 2).mean()
            aux_losses['aux_body_pos_loss'] = body_pos_loss
            aux_losses_weighted['aux_body_pos_loss'] = body_pos_loss * w_pos
        return aux_losses, aux_losses_weighted

    def _compute_reset(self):
        cur_ref_motion_times = self._cur_ref_motion_times
        ref_motion_lengths = self._motion_lib._motion_lengths.to(self.device)[self._reset_ref_motion_ids]

        old_reset_buf = self.reset_buf.clone()
        old_terminate_buf = self._terminate_buf.clone()
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos, self.max_episode_length,
                                                                           self._enable_early_termination, self._termination_heights,
                                                                           cur_ref_motion_times, ref_motion_lengths)
        reset_mask = old_reset_buf == 1
        if torch.any(reset_mask):
            self.reset_buf[reset_mask] = 1
            self._terminate_buf[reset_mask] = old_terminate_buf[reset_mask]
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._rigid_body_pos[env_ids] = rb_pos
        self._rigid_body_rot[env_ids] = rb_rot
        self._rigid_body_vel[env_ids] = 0
        self._rigid_body_ang_vel[env_ids] = 0
        self._reset_rb_pos = rb_pos.clone()
        self._reset_rb_rot = rb_rot.clone()

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def render_vis(self, init=False):
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    # ZL: removing the base rotation for SMPL model
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    return quat_mul(quat, base_rot.repeat(quat.shape[0], 1))


@torch.jit.script
def compute_humanoid_observations_imitation(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, motion_bodies, local_root_obs, root_height_obs):
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
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
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
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])
    # target body rot   [N, 6xB]
    target_rel_body_rot = quat_mul(quat_conjugate(body_rot), target_rot)
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4)).view(target_rel_body_rot.shape[0], -1)


    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_root_rot_obs, target_rel_2d_pos, target_rel_heading_vec, target_rel_dof_pos, target_rel_body_pos, target_rel_body_rot_obs, motion_bodies), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations_imitation_jpos(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, body_vel, body_ang_vel, motion_bodies, local_root_obs, root_height_obs):
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
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
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
    flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1], target_rel_body_pos.shape[2])
    flat_target_rel_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_target_rel_body_pos)
    target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0], target_rel_body_pos.shape[1] * target_rel_body_pos.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, dof_vel,
                     target_rel_root_h, target_rel_2d_pos, target_rel_body_pos, motion_bodies), dim=-1)
    return obs
    

@torch.jit.script
def compute_humanoid_reward(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, target_dof_pos, target_dof_vel, body_vel, body_ang_vel, dof_obs_size, dof_offsets, body_pos_weights, reward_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, List[int], Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor, str]
    
    k_dof, k_vel, k_pos, k_rot = reward_specs['k_dof'], reward_specs['k_vel'], reward_specs['k_pos'], reward_specs['k_rot']
    w_dof, w_vel, w_pos, w_rot = reward_specs['w_dof'], reward_specs['w_vel'], reward_specs['w_pos'], reward_specs['w_rot']
    
    # dof rot reward
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    target_dof_obs = dof_to_obs(target_dof_pos, dof_obs_size, dof_offsets)
    diff_dof_obs = dof_obs - target_dof_obs
    diff_dof_obs_dist = (diff_dof_obs ** 2).mean(dim=-1)
    dof_reward = torch.exp(-k_dof * diff_dof_obs_dist)

    # velocity reward
    diff_dof_vel = target_dof_vel - dof_vel
    diff_dof_vel_dist = (diff_dof_vel ** 2).mean(dim=-1)
    vel_reward = torch.exp(-k_vel * diff_dof_vel_dist)

    # body pos reward
    diff_body_pos = target_pos - body_pos
    diff_body_pos = diff_body_pos * body_pos_weights[:, None]
    diff_body_pos_dist = (diff_body_pos ** 2).mean(dim=-1).mean(dim=-1)
    body_pos_reward = torch.exp(-k_pos * diff_body_pos_dist)

    # body rot reward
    diff_body_rot = quat_mul(target_rot, quat_conjugate(body_rot))
    diff_body_rot_angle = torch_utils.quat_to_angle_axis(diff_body_rot)[0]
    diff_body_rot_angle_dist = (diff_body_rot_angle ** 2).mean(dim=-1)
    body_rot_reward = torch.exp(-k_rot * diff_body_rot_angle_dist)

    # reward = dof_reward * vel_reward * body_pos_reward * body_rot_reward
    reward = w_dof * dof_reward + w_vel * vel_reward + w_pos * body_pos_reward + w_rot * body_rot_reward
    sub_rewards = torch.stack([dof_reward, vel_reward, body_pos_reward, body_rot_reward], dim=-1)
    sub_rewards_names = 'dof_reward,vel_reward,body_pos_reward,body_rot_reward'
    return reward, sub_rewards, sub_rewards_names


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, cur_ref_motion_times, ref_motion_lengths):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    
    # enable_early_termination = False
    if (enable_early_termination):
        # masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0
        # fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = fall_height

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reach_max_length = progress_buf >= max_episode_length - 1
    reach_max_dur = cur_ref_motion_times >= ref_motion_lengths
    reset_cond = torch.logical_or(reach_max_length, reach_max_dur)
    reset = torch.where(reset_cond, torch.ones_like(reset_buf), terminated)

    return reset, terminated
