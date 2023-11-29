# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import yaml
import joblib
import torch
from utils import torch_utils

from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.core.rotation3d import *


USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
    def __getattr__(self, string):
        out = getattr(super().__getattribute__('obj'), string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 key_body_ids, device, clean_up=False):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._load_motions(motion_file)

        motions = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float()
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float()
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float()
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()

        self.generate_length_starts()

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        if clean_up:
            del self._motions
            del self._motion_aa

        return

    def generate_length_starts(self):
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

    def merge_multiple_motion_libs(self, motion_lib_arr):
        keys = ['gts', 'grs', 'lrs', 'grvs', 'gravs', 'dvs', '_motion_weights',
                '_motion_lengths', '_motion_num_frames', '_motion_dt', '_motion_fps', '_motion_bodies', '_motion_body_idx', '_motion_body_scales', '_motion_min_verts_h', '_motion_seq_ids', '_motion_seq_names', '_motion_kp2d', '_motion_cam_proj']
        
        for key in keys:
            if key not in self.__dict__ or self.__dict__[key] is None:
                continue
            
            if key == '_motion_seq_names':
                for mlib in motion_lib_arr:
                    self.__dict__[key].extend(mlib.__dict__[key])
            else:
                arr = [self.__dict__[key]] + [mlib.__dict__[key] for mlib in motion_lib_arr]
                self.__dict__[key] = torch.cat(arr, dim=0)
        
        self.generate_length_starts()
        self._motion_weights = self._motion_weights / sum(self._motion_weights)
        self.motion_ids = torch.arange(len(self._motion_lengths), dtype=torch.long, device=self._device)

    def num_motions(self):
        return len(self.motion_ids)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n, weights_from_lenth=True):
        if weights_from_lenth:
            motion_weights = self._motion_lengths / sum(self._motion_lengths)
        else:
            motion_weights = self._motion_weights
        motion_ids = torch.multinomial(motion_weights, num_samples=n, replacement=True)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None, motion_time_range=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len = torch.clamp_min(motion_len - truncate_time, 0)
        
        if motion_time_range is not None:
            start, end = motion_time_range
            if start is None:
                start = 0
            start = torch.ones_like(motion_len) * start
            if end is None:
                end = motion_len
            else:
                end = torch.ones_like(motion_len) * end
            motion_time = start + (end - start) * phase
        else:
            motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times, return_rigid_body=False, return_body_shape=False, num_motion_cycles=0, motion_cycle_len=None, device=None, adjust_height=False, ground_tolerance=0.0, return_kp2d=False):

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]
        
        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64


        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        if motion_cycle_len is not None:
            fr_start = self.length_starts[motion_ids]
            fr_end = fr_start + motion_cycle_len
            root_start_pos = self.gts[fr_start, 0]
            root_start_rot = self.grs[fr_start, 0]
            root_end_pos = self.gts[fr_end, 0]
            root_end_rot = self.grs[fr_end, 0]
            cycle_trans = root_end_pos - root_start_pos
            new_trans = cycle_trans * num_motion_cycles[:, None]
            new_trans[:, 2] = 0

            root_pos += new_trans
            key_pos += new_trans[:, None, :]

        if adjust_height:
            min_vh = self._motion_min_verts_h[motion_ids] - ground_tolerance
            root_pos[..., 2] -= min_vh
            key_pos[..., 2] -= min_vh[:, None]

        res = (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos)
    
        if return_rigid_body:
            rb_pos0 = self.gts[f0l]
            rb_pos1 = self.gts[f1l]
            rb_pos = (1.0 - blend_exp) * rb_pos0 + blend_exp * rb_pos1

            rb_rot0 = self.grs[f0l]
            rb_rot1 = self.grs[f1l]
            rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)

            if motion_cycle_len is not None:
                rb_pos += new_trans[:, None, :]

            if adjust_height:
                rb_pos[..., 2] -= min_vh[:, None]

            res += (rb_pos, rb_rot)

        if return_body_shape:
            res += (self._motion_bodies[motion_ids])

        if return_kp2d:
            blend_round = torch.round(blend_exp)
            kp2d0 = self._motion_kp2d[f0l]
            kp2d1 = self._motion_kp2d[f1l]
            kp2d = (1.0 - blend_round) * kp2d0 + blend_round * kp2d1
            
            cam_proj0 = self._motion_cam_proj[f0l]
            cam_proj1 = self._motion_cam_proj[f1l]
            cam_proj = (1.0 - blend_round) * cam_proj0 + blend_round * cam_proj1

            res += (kp2d, cam_proj)
        
        if device is not None and res[0].device != device:
            res = tuple([x.to(device) for x in res])

        return res

    def get_all_rb_pos(self, motion_ids, adjust_height=False, ground_tolerance=0.0):
        fr_start = self.length_starts[motion_ids]
        fr_end = fr_start + self._motion_num_frames[motion_ids]
        rb_pos = self.gts[fr_start:fr_end].clone()
        if adjust_height:
            min_vh = self._motion_min_verts_h[motion_ids] - ground_tolerance
            rb_pos[..., 2] -= min_vh
        return rb_pos

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_bodies = []
        self._motion_body_scales = []
        self._motion_body_idx = []
        self._motion_aa = []
        self._motion_kp2d = []
        self._motion_cam_proj = []
        self._motion_min_verts_h = []
        self._motion_seq_names = []
        self._motion_seq_ids = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            # normal string file name
            if (isinstance(curr_file, str)):
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
                motion_file_data = np.load(curr_file, allow_pickle=True).item()
                curr_motion = SkeletonMotion.from_dict(motion_file_data)
                self._motion_aa.append(torch.zeros(72, device=self._device, dtype=torch.float32))
                self._motion_bodies.append(torch.zeros(17, device=self._device, dtype=torch.float32))
                self._motion_body_scales.append(1.0)
                self._motion_body_idx.append(0)
                self._motion_min_verts_h.append(0.0)
                self._motion_files.append(curr_file)
            # data dict
            elif (isinstance(curr_file, dict)):
                motion_file_data = curr_file
                curr_motion = SkeletonMotion.from_dict(curr_file)
                if "beta" in motion_file_data:
                    beta, gender, pose_aa, min_verts_h = motion_file_data['beta'], motion_file_data['gender'], motion_file_data['pose_aa'], motion_file_data['min_verts_h']

                    if isinstance(gender, bytes):
                        gender = gender.decode("utf-8")
                    if gender == "neutral":
                        gender = [0]
                    elif gender == "male":
                        gender = [1]
                    elif gender == "female":
                        gender = [2]
                    else:
                        raise Exception("Gender Not Supported!!")
                    self._motion_aa.append(torch.tensor(pose_aa, device=self._device, dtype=torch.float32))
                    self._motion_bodies.append(torch.tensor(np.concatenate((gender, beta)), device=self._device, dtype=torch.float32))
                    self._motion_body_scales.append(motion_file_data['body_scale'])
                    self._motion_body_idx.append(motion_file_data['beta_idx'])
                    self._motion_min_verts_h.append(min_verts_h)
                    self._motion_seq_ids.append(motion_file_data['seq_idx'])
                    self._motion_seq_names.append(motion_file_data['seq_name'])
                    if 'kp2d' in motion_file_data:
                        self._motion_kp2d.append(torch.tensor(motion_file_data['kp2d'], device=self._device, dtype=torch.float32))
                        self._motion_cam_proj.append(torch.tensor(motion_file_data['cam_proj'], device=self._device, dtype=torch.float32))
                else:
                    raise Exception("No beta in motion file!")

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(self._motion_bodies, dim=0)
        self._motion_body_scales = torch.tensor(self._motion_body_scales, device=self._device, dtype=torch.float32)
        self._motion_body_idx = torch.tensor(self._motion_body_idx, device=self._device)
        self._motion_kp2d = torch.cat(self._motion_kp2d, dim=0) if len(self._motion_kp2d) > 0 else None
        self._motion_cam_proj = torch.cat(self._motion_cam_proj, dim=0) if len(self._motion_cam_proj) > 0 else None
        self._motion_min_verts_h = torch.tensor(self._motion_min_verts_h, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        self._motion_seq_ids = torch.tensor(self._motion_seq_ids, device=self._device)

        return

    def _fetch_motion_files(self, motion_file):

        if isinstance(motion_file, dict):

            motion_files = list(motion_file.values())
            motion_weights = [1/len(motion_files)] * len(motion_files)

        else:
            ext = os.path.splitext(motion_file)[1]
            if (ext == ".yaml"):
                dir_name = os.path.dirname(motion_file)
                motion_files = []
                motion_weights = []

                with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                    motion_config = yaml.load(f, Loader=yaml.SafeLoader)

                motion_list = motion_config['motions']
                for motion_entry in motion_list:
                    curr_file = motion_entry['file']
                    curr_weight = motion_entry['weight']
                    assert(curr_weight >= 0)

                    curr_file = os.path.join(dir_name, curr_file)
                    motion_weights.append(curr_weight)
                    motion_files.append(curr_file)

            elif (ext == ".pkl"):
                motion_data = joblib.load(motion_file)
                motion_files = list(motion_data.values())
                motion_weights = [1/len(motion_files)] * len(motion_files)


            else:
                motion_files = [motion_file]
                motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel

            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel