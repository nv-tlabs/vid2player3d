from rl_games.algos_torch import network_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from isaacgym.torch_utils import *
import torch
import torch.nn as nn
import numpy as np

from .running_norm import RunningNorm
from utils import torch_utils
from utils.torch_transform import heading_to_vec, rotation_matrix_to_angle_axis, rotation_matrix_to_quaternion, rot6d_to_rotmat
from utils.hybrik import batch_inverse_kinematics_transform_naive, batch_inverse_kinematics_transform
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES as smpl_joint_names


DISC_LOGIT_INIT_SCALE = 1.0

mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]
smpl_2_mujoco = [smpl_joint_names.index(q) for q in mujoco_joint_names]
mujoco_2_smpl = [mujoco_joint_names.index(q) for q in smpl_joint_names]


class ImitatorBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.context_padding = params.get('context_padding', 8)
            self.humanoid_obs_dim = params.get('humanoid_obs_dim', 734)
            self.residual_action = params.get('residual_action', True)
            self.use_running_obs = params.get('use_running_obs', False)
            self.running_obs_type = params.get('running_obs_type', 'rl_game')
            self.use_ik = params.get('use_ik', False)
            self.ik_type = params.get('ik_type', 'optimized')
            self.ik_ignore_outlier = params.get('ik_ignore_outlier', False)
            self.kinematic_pretrained = params.get('kinematic_pretrained', False)
            
            self.smpl_rest_joints = kwargs['smpl_rest_joints']
            self.smpl_parents = kwargs['smpl_parents']
            self.smpl_children = kwargs['smpl_children']

            kwargs['input_shape'] = (self.humanoid_obs_dim,)
            super().__init__(params, **kwargs)

            if self.use_running_obs:
                if self.running_obs_type == 'rl_game':
                    self.running_obs = RunningMeanStd((self.humanoid_obs_dim,))
                else:
                    self.running_obs = RunningNorm(self.humanoid_obs_dim)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            return

        def load(self, params):
            super().load(params)
            return

        def setup_env_named_dims(self, obs_names, obs_shapes, obs_dims, context_names, context_shapes, context_dims):
            self.obs_names = obs_names
            self.obs_shapes = obs_shapes
            self.obs_dims = obs_dims
            self.context_names = context_names
            self.context_shapes = context_shapes
            self.context_dims = context_dims
            return

        def perform_ik(self, body_pos, body_rot, dof_pos, phis=None, thetas=None, env_id=None):
            body_pos_flat = body_pos.view(-1, *body_pos.shape[2:])
            smpl_body_pos = body_pos_flat[:, mujoco_2_smpl]
            rest_body_pos = self.smpl_rest_joints[env_id] if env_id is not None else self.smpl_rest_joints
            rest_body_pos = rest_body_pos.repeat_interleave(body_pos.shape[1], dim=0)

            # phis
            if phis is None:
                phis = torch.tensor([1.0, 0.0], device=body_pos.device).expand(smpl_body_pos.shape[0], 23, -1)
            else:
                phis = phis.view(smpl_body_pos.shape[0], 23, 2)
                phis += torch.tensor([1.0, 0.0], device=phis.device)
            # leaf thetas
            if thetas is None:
                leaf_thetas = torch.eye(3, device=body_pos.device).expand(smpl_body_pos.shape[0], 5, -1, -1)
            else:
                new_thetas = thetas.view(smpl_body_pos.shape[0], 5, 6) + torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=thetas.device)
                leaf_thetas = rot6d_to_rotmat(new_thetas)

            root_diff = rest_body_pos[:, [0]] - smpl_body_pos[:, [0]]
            smpl_body_pos += root_diff
            if self.ik_type == 'optimized':
                smpl_rot_mats, global_rot_mat, global_body_pos = batch_inverse_kinematics_transform(smpl_body_pos, None, phis, rest_body_pos, self.smpl_children, self.smpl_parents, leaf_thetas, self.ik_ignore_outlier)   # 0.012s - 0.015s
            else:
                smpl_rot_mats, global_rot_mat = batch_inverse_kinematics_transform_naive(smpl_body_pos, None, phis, rest_body_pos, self.smpl_children, self.smpl_parents, leaf_thetas)     # 0.007s - 0.010s
                global_body_pos = None
            rot_mats = smpl_rot_mats[:, smpl_2_mujoco].view(*body_pos.shape[:2], -1, 3, 3)
            ik_dof_pos = rotation_matrix_to_angle_axis(rot_mats)    # 0.002s
            ik_body_rot = rotation_matrix_to_quaternion(global_rot_mat.contiguous())[..., [1, 2, 3, 0]]    # 0.001s
            ik_body_rot = ik_body_rot[:, smpl_2_mujoco].view(*body_pos.shape[:2], -1, 4)
            if global_body_pos is not None:
                ik_body_pos = global_body_pos - root_diff
                ik_body_pos = ik_body_pos[:, smpl_2_mujoco].view(*body_pos.shape[:2], -1, 3)
            else:
                ik_body_pos = None

            recon_err = None
            return ik_dof_pos, ik_body_rot, ik_body_pos, recon_err

        def forward_context(self, context_feat, mask, env_id=None, flatten=False):
            self.context = dict()
            context_chunks = torch.split(context_feat, self.context_dims, dim=-1)
            for name, shape, chunk in zip(self.context_names, self.context_shapes, context_chunks):
                self.context[name] = chunk.view(chunk.shape[:2] + shape)

            if self.use_ik:
                self.context['ik_dof_pos'], self.context['ik_body_rot'], self.context['ik_body_pos'], self.context['ik_err'] = self.perform_ik(self.context['body_pos'], self.context['body_rot'], self.context['dof_pos'], self.context['ik_phis'], self.context['ik_thetas'], env_id)
                self.context['dof_pos'] = self.context['ik_dof_pos'][..., 1:, :].reshape(*self.context['ik_dof_pos'].shape[:2], -1)
                self.context['body_rot'] = self.context['ik_body_rot']
                self.context['body_pos'] = self.context['ik_body_pos']

            context_names = self.context_names + (['ik_dof_pos', 'ik_body_pos'] if self.use_ik else [])
            for name in context_names:
                if self.context[name] is None:
                    continue
                if flatten:
                    self.context[name] = self.context[name][:, self.context_padding:-self.context_padding].reshape(-1, *self.context[name].shape[2:])
                else:
                    self.context[name] = self.context[name][:, self.context_padding:-self.context_padding + 1]

            return self.context

        def obtain_cur_context(self, t):
            if t is None:
                cur_context = self.context
            else:
                cur_context = {name: self.context[name][:, t] for name in self.context_names}
            return cur_context

        def compute_humanoid_obs(self, obs_feat, cur_context):
            body_pos = obs_feat['body_pos']
            body_rot = obs_feat['body_rot']
            body_vel = obs_feat['body_vel']
            body_ang_vel = obs_feat['body_ang_vel']
            dof_pos = obs_feat['dof_pos']
            dof_vel = obs_feat['dof_vel']
            motion_bodies = obs_feat['motion_bodies']
            target_pos = cur_context['body_pos']
            target_rot = cur_context['body_rot']
            target_dof_pos = cur_context['dof_pos']
            obs = compute_humanoid_observations_imitation(body_pos, body_rot, target_pos, target_rot, dof_pos, dof_vel, 
                                                          target_dof_pos, body_vel, body_ang_vel, motion_bodies, True, True)
            return obs

        def preprocess_input(self, obs_dict):
            if self.training:
                flatten = True
                assert 'context_feat' in obs_dict
                context_feat = obs_dict['context_feat']
                mask = obs_dict['context_mask']
                self.forward_context(context_feat, mask, env_id=obs_dict['env_id'], flatten=flatten)
            else:
                flatten = False

            obs_feat = dict()
            obs_chunks = torch.split(obs_dict['obs'], self.obs_dims, dim=-1)
            for name, shape, chunk in zip(self.obs_names, self.obs_shapes, obs_chunks):
                if flatten:
                    obs_feat[name] = chunk.view(np.prod(chunk.shape[:2]), *shape)
                else:
                    obs_feat[name] = chunk.view(chunk.shape[:1] + shape)

            t = obs_dict.get('t', None)
            cur_context = self.obtain_cur_context(t)

            obs_dict['human_obs'] = self.compute_humanoid_obs(obs_feat, cur_context)
            obs_dict['cur_context'] = cur_context
            if self.use_running_obs:
                obs_dict['obs_processed'] = self.running_obs(obs_dict['human_obs'])
            else:
                obs_dict['obs_processed'] = obs_dict['human_obs']
            return

        def forward(self, obs_dict):

            actor_outputs = self.eval_actor(obs_dict)
            value = self.eval_critic(obs_dict)

            extra = {'context': obs_dict['cur_context']}

            output = actor_outputs + (value, None, extra)

            return output

        def eval_actor(self, obs_dict):
            if 'obs_processed' not in obs_dict:
                self.preprocess_input(obs_dict)
            obs = obs_dict['obs_processed']

            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                if self.residual_action:
                    target_dof_pos = obs_dict['cur_context']['dof_pos']
                    mu[:, :target_dof_pos.shape[-1]] += target_dof_pos

                return mu, sigma
            return

        def eval_critic(self, obs_dict):
            if 'obs_processed' not in obs_dict:
                self.preprocess_input(obs_dict)
            obs = obs_dict['obs_processed']

            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = ImitatorBuilder.Network(self.params, **kwargs)
        return net


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