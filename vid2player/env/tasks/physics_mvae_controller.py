from players.mvae_player import MVAEPlayer
from env.utils.player_builder import PlayerBuilder
from utils.tennis_ball_out_estimator import TennisBallOutEstimator
from utils.common import AverageMeter
from utils.torch_transform import quat_to_rot6d
from utils import torch_utils

import torch
import math
import pdb


class PhysicsMVAEController():
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.cfg_v2p = self.cfg['env']['vid2player']
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self._max_episode_length = self.cfg["env"]["episodeLength"]
        self._enable_early_termination = self.cfg["env"].get("enableEarlyTermination", False)
        self._is_train = self.cfg['env']['is_train']
        
        # from BaseTask
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.headless = cfg["headless"]
        self.num_envs = cfg["env"]["numEnvs"]
        self.create_sim()

        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]
        self._epoch_num = 0

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._sub_rewards = None
        self._sub_rewards_names = None

        self._has_init = False
        self._racket_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._racket_normal = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._reward_scales = self.cfg_v2p.get('reward_scales', {}).copy()
        # racket body has already been fliped to be the last body
        self._num_humanoid_bodies = 24
        self._racket_body_id = 24
        self._head_body_id = 13

        self._obs_ball_traj_length = self.cfg_v2p.get('obs_ball_traj_length', 100)
        self._ball_traj = torch.zeros((self.num_envs, 100, 3), device=self.device, dtype=torch.float32)
        self._ball_obs = torch.zeros((self.num_envs, self._obs_ball_traj_length, 3), 
            device=self.device, dtype=torch.float32)
        self._bounce_in = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # estimate bounce for stats
        self._est_bounce_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._est_bounce_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._est_bounce_in = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._est_max_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        if not self.cfg_v2p.get('dual_mode'):
            self._ball_out_estimator = TennisBallOutEstimator(
                self.cfg_v2p.get('ball_traj_out_x_file', 'vid2player/data/ball_traj_out_x_v0.npy'),
                self.cfg_v2p.get('ball_traj_out_y_file', 'vid2player/data/ball_traj_out_y_v0.npy'),
            )
        if self.cfg_v2p.get('court_min'):
            self._court_min = torch.FloatTensor(self.cfg_v2p['court_min']).to(self.device)
            self._court_max = torch.FloatTensor(self.cfg_v2p['court_max']).to(self.device)
            self._court_range = self._court_max - self._court_min
            print("Court range:", self._court_min, self._court_max)

        # target
        self._tar_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._tar_time_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._tar_action = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64) # 1 swing 0 recovery
        self._target_bounce_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._target_bounce_pos[:] = torch.FloatTensor([0, 10, 0])
        if self.cfg_v2p.get('use_random_ball_target', False) == 'continuous':
            self._target_bounce_min = torch.FloatTensor([-3, 9, 0]).to(self.device)
            self._target_bounce_max = torch.FloatTensor([3, 11, 0]).to(self.device)

        # all envs start with reaction task
        self._reset_reaction_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_recovery_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._num_reset_reaction = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._num_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self._sub_rewards = [None] * self.num_envs
        self._sub_rewards_names = None
        self._mvae_actions = None
        self._res_dof_actions = None
        if not self._is_train or self.cfg_v2p.get('random_walk_in_recovery', False):
            self._mvae_actions_random = torch.zeros((self.num_envs, self._num_mvae_action), device=self.device, dtype=torch.float32)
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
    
    def get_action_size(self):
        return self._num_actions
    
    def get_actor_obs_size(self):
        return self._num_actor_obs
    
    def create_sim(self):
        smpl_beta_dict = {}
        smpl_beta_dict['djokovic'] = [-0.9807,  1.4050, -0.4144,  1.4028, -1.3299,  
                2.0045, -1.3108,  0.7475, -0.0924, -0.3262]
        smpl_beta_dict['federer'] = [-0.6303,  1.1747, -0.3463,  1.0915, -1.0501,  
                1.7888, -1.1762,  0.6059, 0.2589, -0.4568]
        smpl_beta_dict['nadal'] = [-0.6278,  1.2620, -0.3143,  0.8561, -0.8136,  
                1.3917, -0.9902,  0.5112,  0.2334, -0.4266]
        
        if self.cfg_v2p.get('dual_mode') == 'different':
            player = self.cfg_v2p['player']
            self.cfg_v2p['smpl_beta'] = [smpl_beta_dict[player[0]], smpl_beta_dict[player[1]]]
            self.betas = torch.FloatTensor(self.cfg_v2p['smpl_beta']).repeat(self.num_envs//2, 1).to(self.device)
        else:
            self.cfg_v2p['smpl_beta'] = [0] * 10
            if self.cfg_v2p.get('player', None) is not None:
                self.cfg_v2p['smpl_beta'] = [smpl_beta_dict[self.cfg_v2p['player']]]
            self.betas = torch.FloatTensor(self.cfg_v2p['smpl_beta']).repeat(self.num_envs, 1).to(self.device)

        self._mvae_player = MVAEPlayer(self.cfg_v2p, num_envs=self.num_envs, 
            is_train=self._is_train, enable_physics=True,
            device=self.device)
        self._smpl = self._mvae_player._smpl

        self._physics_player = None
        self._physics_player = PlayerBuilder(self, self.cfg).build_player()
        self._physics_player.task._smpl = self._smpl
        self._physics_player.task._mvae_player = self._mvae_player
        self._physics_player.task._controller = self

        self._num_actor_obs = 3 + 3 + 24*3 + 24*6 + 3
        self._num_mvae_action = self._num_actions = 32
        self._num_res_dof_action = 0
        if self.cfg_v2p.get('add_residual_dof'):
                self._num_res_dof_action += 3
        self._num_actions += self._num_res_dof_action
        if self.cfg_v2p.get('add_residual_root'):
            self._num_actions += 3
    
        self.cfg["env"]["numActions"] = self.get_action_size()
        self.cfg["env"]["numObservations"] =  self.get_actor_obs_size() + self.get_task_obs_size()
    
    def get_task_obs_size(self):
        self._num_task_obs = 3 * self.cfg_v2p.get('obs_ball_traj_length', 100)

        if self.cfg_v2p.get('use_random_ball_target', False):
            self._num_task_obs += 2
        return self._num_task_obs
    
    def reset(self, env_ids=None):
        if env_ids is None:
            if self._has_init and self.num_envs > 1: return
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
    
    def _reset_envs(self, env_ids):
        reset_humanoid_env_ids = env_ids
        # envs that need to reset task(ball)
        reset_reaction_env_ids = self._reset_reaction_buf.nonzero(as_tuple=False).flatten() # include reset_humanoid_env_ids
        reset_recovery_env_ids = self._reset_recovery_buf.nonzero(as_tuple=False).flatten()
        reset_all_env_ids = (self._reset_reaction_buf + self._reset_recovery_buf).nonzero(as_tuple=False).flatten()
        
        if (len(reset_humanoid_env_ids) > 0):
            self._reset_env_tensors(reset_humanoid_env_ids)
        if (len(reset_humanoid_env_ids) > 0):
            self._mvae_player.reset(reset_humanoid_env_ids)
            self._num_reset[reset_humanoid_env_ids] += 1
        if len(reset_humanoid_env_ids) > 0 or len(reset_reaction_env_ids) > 0:
            new_traj = self._physics_player.task.reset(reset_humanoid_env_ids, reset_reaction_env_ids)
            if not self.cfg_v2p.get('use_history_ball_obs', False):
                self._ball_traj[reset_reaction_env_ids] = new_traj.to(self.device)
            if not self.headless and len(reset_humanoid_env_ids) == self.num_envs:
                self._physics_player.task.render_vis(init=True)
        if (len(reset_humanoid_env_ids) > 0):
            self._update_state() # Setting the right fields to compute obs

        if len(reset_recovery_env_ids) > 0:
            self._reset_recovery_tasks(reset_recovery_env_ids)
        if len(reset_reaction_env_ids) > 0:
            self._reset_reaction_tasks(reset_reaction_env_ids, reset_humanoid_env_ids)
        if len(reset_all_env_ids) > 0:
            self._compute_observations(reset_all_env_ids)
        
        self._has_init = True

    def _reset_env_tensors(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._reset_reaction_buf[env_ids] = 0
        self._reset_recovery_buf[env_ids] = 0
        self._num_reset_reaction[env_ids] = 0
        self._distance[env_ids] = 0

    def _reset_reaction_tasks(self, env_ids, humanoid_env_ids=None):
        if self.cfg_v2p.get('use_history_ball_obs'):
            self._ball_obs[env_ids] = self._ball_pos[env_ids].view(-1, 1, 3).repeat(1, self._obs_ball_traj_length, 1)
        self._tar_time[env_ids] = 0
        self._tar_action[env_ids] = 1
        self._num_reset_reaction[env_ids] += 1
        self._bounce_in[env_ids] = 0
        self._est_bounce_pos[env_ids, :] = 0
        self._est_bounce_time[env_ids] = 0
        self._est_bounce_in[env_ids] = 0
        self._est_max_height[env_ids] = 0
        self._mvae_player._swing_type_cycle[env_ids] = -1

        self._tar_time_total[env_ids] = self.cfg_v2p['reset_reaction_nframes'] + \
            torch.randint(-5, 5, (len(env_ids),), device=self.device)

        if self.cfg_v2p.get('use_random_ball_target'):
            if self.cfg_v2p['use_random_ball_target'] == 'continuous':
                # use same target for the envs to be reset
                self._target_bounce_pos[env_ids] = torch.rand((3,), device=self.device) \
                    * (self._target_bounce_max - self._target_bounce_min) + self._target_bounce_min
            else:
                rand_seed = torch.rand((len(env_ids)))
                bounce_left = rand_seed < 0.33
                bounce_right = rand_seed > 0.67
                bounce_middle = ~bounce_left & ~bounce_right
                self._target_bounce_pos[env_ids[bounce_left]] = torch.FloatTensor([-3, 10, 0]).to(self.device)
                self._target_bounce_pos[env_ids[bounce_middle]] = torch.FloatTensor([0, 10, 0]).to(self.device)
                self._target_bounce_pos[env_ids[bounce_right]] = torch.FloatTensor([3, 10, 0]).to(self.device)
        
    def _reset_recovery_tasks(self, env_ids):
        self._tar_action[env_ids] = 0
        self._physics_player.task._has_bounce[env_ids] = 0
        self._physics_player.task._bounce_pos[env_ids] = 0
        
    def pre_physics_step(self, actions):
        self._actions = actions.clone()
        self._mvae_actions = actions[:, :self._num_mvae_action].clone() * \
            self.cfg_v2p.get('vae_action_scale', 1.0)
        
        if self.cfg_v2p.get('random_walk_in_recovery', False):
            in_recovery = self._tar_action == 0
            num_in_recovery = in_recovery.sum()
            self._mvae_actions_random[:num_in_recovery].normal_(0, 1)
            self._mvae_actions_random[:num_in_recovery] = torch.clamp(self._mvae_actions_random[:num_in_recovery], -5, 5)
            self._mvae_actions[in_recovery] = self._mvae_actions_random[:num_in_recovery]

        self._res_dof_actions = torch.empty(0)
        if self.cfg_v2p.get('add_residual_dof'):
            self._res_dof_actions = actions[:, self._num_mvae_action:self._num_mvae_action+self._num_res_dof_action].clone() \
                * self.cfg_v2p.get('residual_dof_scale', 0.1)
        self._mvae_player.step(self._mvae_actions, self._res_dof_actions)
        if self.cfg_v2p.get('add_residual_root'):
            self._res_root_actions = actions[:, self._num_mvae_action + self._num_res_dof_action:
                self._num_mvae_action + self._num_res_dof_action + 3].clone() \
                * self.cfg_v2p.get('residual_root_scale', 0.02)

        self._physics_player.task.post_mvae_step()
    
    def _update_state(self):
        self._root_pos = self._physics_player.task._root_pos
        self._root_vel = self._physics_player.task._root_vel
        if not self._is_train:
            self._joint_rot = self._physics_player.task._joint_rot

        self._racket_pos = self._physics_player.task._racket_pos
        self._racket_vel = self._physics_player.task._racket_vel
        self._racket_normal = self._physics_player.task._racket_normal

        self._ball_pos = self._physics_player.task._ball_pos
        self._ball_vel = self._physics_player.task._ball_vel
        self._ball_vspin = self._physics_player.task._ball_vspin

        court_min = [-4.11, 0]
        court_max = [4.11, 11.89]
        serve_min = [0., 0]
        serve_max = [4.11, 6.4]
        update_true_bounce = (self._tar_action == 0) & self._physics_player.task._has_bounce_now
        bounce_pos = self._physics_player.task._bounce_pos[update_true_bounce]
        self._bounce_in[update_true_bounce] = \
            (bounce_pos[:, 0] > court_min[0]) & \
            (bounce_pos[:, 0] < court_max[0]) & \
            (bounce_pos[:, 1] > court_min[1]) & \
            (bounce_pos[:, 1] < court_max[1])

        # estimate bounce position
        has_contact_now = self._physics_player.task._has_racket_ball_contact_now
        if not self.cfg_v2p.get('dual_mode') and has_contact_now.sum() > 0:
            has_valid_contact, bounce_pos, bounce_time, max_height = self._ball_out_estimator.estimate(
                self._physics_player.task._ball_root_states[has_contact_now])
            if has_valid_contact.sum() > 0:
                env_ids = has_contact_now.nonzero(as_tuple=False).flatten()[has_valid_contact]
                self._est_bounce_pos[env_ids, :2] = bounce_pos
                self._est_bounce_time[env_ids] = bounce_time
                self._est_max_height[env_ids] = max_height

                self._est_bounce_in[env_ids] = \
                    (self._est_bounce_pos[env_ids, 0] > court_min[0]) & \
                    (self._est_bounce_pos[env_ids, 0] < court_max[0]) & \
                    (self._est_bounce_pos[env_ids, 1] > court_min[1]) & \
                    (self._est_bounce_pos[env_ids, 1] < court_max[1])
                
        self._phase_pred = self._mvae_player._phase_pred

    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        actor_obs = self._compute_actor_obs(env_ids)
        if torch.isnan(actor_obs).any():
            print("Found NAN in actor obersavations")
            pdb.set_trace(())
        
        task_obs = self._compute_task_obs(env_ids)
        if torch.isnan(task_obs).any():
            print("Found NAN in task obersavations")
            pdb.set_trace()
        
        obs = torch.cat([actor_obs, task_obs], dim=-1)
        self.obs_buf[env_ids] = obs
    
    def _compute_actor_obs(self, env_ids):
        player = self._physics_player.task
        actor_obs = torch.cat([
            player._root_pos[env_ids], 
            player._root_vel[env_ids],
            (player._rigid_body_pos[env_ids, 1:] - player._root_pos[env_ids].unsqueeze(-2)).view(-1, 24*3),
            quat_to_rot6d(player._rigid_body_rot[env_ids, :self._num_humanoid_bodies].view(-1, 4)).view(-1, 24*6),
            player._racket_normal[env_ids],
        ], dim=-1)
        return actor_obs

    def _compute_task_obs(self, env_ids):
        self._ball_obs[env_ids] = self._ball_obs[env_ids].roll(-1, dims=1)
        self._ball_obs[env_ids, -1] = self._ball_pos[env_ids].clone() # Last is the latest

        if self.cfg_v2p.get('use_history_ball_obs', False):
            ball_obs = self._ball_obs[env_ids]
        else:
            ball_obs = self._ball_traj[env_ids, :self._obs_ball_traj_length]

        # relative to current racket
        task_obs = ball_obs - self._physics_player.task._rigid_body_pos[env_ids, self._racket_body_id].unsqueeze(-2)

        if self.cfg_v2p.get('use_random_ball_target', False):
            target = self._target_bounce_pos[env_ids, :2] - self._physics_player.task._root_pos[env_ids, :2]
            task_obs = torch.cat([task_obs.view(len(env_ids), -1), target], dim=-1)
        
        return task_obs.view(len(env_ids), -1)
    
    def physics_step(self):
        self._physics_player.run_one_step()        

        self._ball_traj = self._ball_traj.roll(-1, dims=1)
        self._ball_traj[:, -1] = 0
    
    def _compute_reward(self, actions):
        reward_type = self.cfg_v2p.get('reward_type', 'return')
        reward_weights = self.cfg_v2p.get('reward_weights', {})

        if reward_type == 'reach':
            self.rew_buf[:], self._sub_rewards, self._sub_rewards_names = compute_reward_reach(
                self._phase_pred, 
                self._tar_action,
                self._racket_pos, 
                self._ball_pos,
                self._mvae_player._swing_type, 
                self._reward_scales,
                reward_weights)
        elif reward_type == 'return':
            self.rew_buf[:], self._sub_rewards, self._sub_rewards_names = compute_reward_return(
                self._phase_pred, 
                self._racket_pos, 
                self._ball_pos,
                self._physics_player.task._has_racket_ball_contact, 
                self._physics_player.task._has_bounce, 
                self._physics_player.task._bounce_pos,
                self._target_bounce_pos, 
                self._mvae_player._swing_type, 
                self._reward_scales,
                reward_weights
                )
        elif reward_type == 'return_w_estimate':
            self.rew_buf[:], self._sub_rewards, self._sub_rewards_names = compute_reward_return_w_estimate(
                self._racket_pos, 
                self._phase_pred, 
                self._mvae_player._swing_type_cycle,
                self._ball_pos,
                self._physics_player.task._has_racket_ball_contact, 
                self._est_bounce_pos, 
                self._est_bounce_time, 
                self._est_bounce_in, 
                self._target_bounce_pos,
                self._reward_scales,
                reward_weights)

    def _compute_reset(self):
        terminated = check_out_of_court(self._root_pos, self._court_min, self._court_max)

        # terminate if player/ball states has NaN
        has_nan = torch.isnan(self.obs_buf).any(dim=1)
        terminated |= has_nan

        self._terminate_buf[:] = terminated
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, 
            torch.ones_like(self.reset_buf), terminated)

        has_contact = self._physics_player.task._has_racket_ball_contact
        self._reset_reaction_buf = self._tar_time == self._tar_time_total 
        self._reset_recovery_buf =  (self._tar_action == 1) & (has_contact | (self._ball_pos[:, 1] < self._root_pos[:, 1] - 1))

        self._compute_stats()

        terminate = self._terminate_buf.bool()
        if self.cfg['env'].get('enableEarlyTermination'):
            # terminate if the player miss the ball
            terminate |= (self._reset_recovery_buf & ~has_contact) | (self._ball_pos[:, 1] < self._root_pos[:, 1]- 1)

            if self.cfg_v2p['reward_type'].startswith('return_w_estimate'):
                terminate |= has_contact & ~self._est_bounce_in
        
        self._terminate_buf[terminate] = 1
        self.reset_buf[terminate] = 1
        self._reset_recovery_buf[terminate] = 0
        self._reset_reaction_buf = self._reset_reaction_buf | self.reset_buf.bool()

    def _compute_stats(self):
        self._distance += (self._root_vel[:, :2]).norm(dim=-1)
                
    def post_physics_step(self):
        self._tar_time += 1
        self.progress_buf += 1

        self._update_state()
        self._compute_reward(self._actions)
        self._compute_observations()
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf
        self.extras["sub_rewards"] = self._sub_rewards
        self.extras["sub_rewards_names"] = self._sub_rewards_names
    
    def step(self, actions):
        self.pre_physics_step(actions)

        self.physics_step()

        self.post_physics_step()
    
    def get_aux_losses(self, model_res_dict):
        aux_loss_specs = self.cfg_v2p.get('aux_loss_specs', dict())
        aux_losses = {}
        aux_losses_weighted = {}

        # default angle to be close to 0
        dof_res = model_res_dict['mus'][:, self._num_mvae_action:self._num_mvae_action+self._num_res_dof_action]
        dof_res_loss = (dof_res ** 2).sum(dim=-1).mean()
        aux_losses['aux_dof_res_loss'] = dof_res_loss
        aux_losses_weighted['aux_dof_res_loss'] = aux_loss_specs.get('dof_res', 0) * dof_res_loss

        return aux_losses, aux_losses_weighted

    def render_vis(self):
        return
    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def check_out_of_court(root_pos, court_min, court_max):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    out_of_court = \
        (root_pos[:, 0] < court_min[0]).logical_or(
        root_pos[:, 1] < court_min[1]).logical_or(
        root_pos[:, 0] > court_max[0]).logical_or(
        root_pos[:, 1] > court_max[1])

    return out_of_court.long()

@torch.jit.script
def compute_reward_reach(
    phase, tar_action, racket_pos, ball_pos, swing_type,
    scales, weights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], Dict[str, float]) -> Tuple[Tensor, Tensor, str]

    mask_reaction = tar_action == 1

    # position
    pos_diff = ball_pos - racket_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    contact_phase = torch.where(
        swing_type == -1,
        torch.ones_like(phase) * 3,
        torch.ones_like(phase) * math.pi,
    )
    phase_diff_rea = phase - contact_phase
    phase_err_rea = phase_diff_rea * phase_diff_rea
    
    pos_reward = mask_reaction * torch.exp(- scales.get('pos', 5.) * pos_err) * \
                torch.exp(- scales.get('phase', 10.) * phase_err_rea)
    
    # all rewards
    reward = pos_reward * weights.get('pos', 1.0)
    sub_rewards = torch.stack([pos_reward], dim=-1)
    sub_rewards_names = 'pos_reward'

    return reward, sub_rewards, sub_rewards_names

@torch.jit.script
def compute_reward_return(
    phase, racket_pos, ball_pos, 
    has_contact, has_bounce, bounce_pos, target_pos, swing_type,
    scales, weights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], Dict[str, float]) -> Tuple[Tensor, Tensor, str]

    w_pos, w_bounce = weights.get('pos', 0.0), weights.get('ball_pos', 0.0)

    # position
    pos_diff = ball_pos - racket_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    contact_phase = torch.where(
        swing_type >= 2,
        torch.ones_like(phase) * 3,
        torch.ones_like(phase) * math.pi,
    )
    phase_diff_rea = phase - contact_phase
    phase_err_rea = phase_diff_rea * phase_diff_rea
    
    pos_reward = ~has_contact * torch.exp(- scales.get('pos', 5.) * pos_err) * \
                torch.exp(- scales.get('phase', 10.) * phase_err_rea)  + \
                has_contact * torch.ones_like(pos_err)

    # outgoing ball pos
    pos_err = torch.where(
        has_bounce,
        torch.sum((bounce_pos - target_pos) **2, dim=-1),
        torch.sum((ball_pos - target_pos) **2, dim=-1),
    )
    # ball_pos_reward = has_contact * torch.exp(-0.05 * pos_err)
    ball_pos_reward = has_contact * torch.clamp((400 - pos_err)/ 400, 0.0, 1.0)

    # all rewards
    reward = w_pos * pos_reward + w_bounce * ball_pos_reward
    sub_rewards = torch.stack([pos_reward, ball_pos_reward], dim=-1)
    sub_rewards_names = 'pos_reward,ball_pos_reward'

    return reward, sub_rewards, sub_rewards_names

@torch.jit.script
def compute_reward_return_w_estimate(
    racket_pos, phase, swing_type, 
    ball_pos, has_contact, bounce_pos, bounce_time, bounce_in, 
    target_pos,
    scales, weights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], Dict[str, float]) -> Tuple[Tensor, Tensor, str]

    w_pos, w_bounce = weights.get('pos', 0.0), weights.get('ball_pos', 0.0)

    # position
    pos_diff = ball_pos - racket_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    # bh contact tends to be earlier
    contact_phase = torch.where(
        (swing_type >= 2),
        torch.ones_like(phase) * 3,
        torch.ones_like(phase) * math.pi,
    )
    phase_diff_rea = phase - contact_phase
    phase_err_rea = phase_diff_rea * phase_diff_rea
    
    pos_reward = ~has_contact * torch.exp(- scales.get('pos', 5.) * pos_err) \
                * torch.exp(- scales.get('phase', 10.) * phase_err_rea) \
                + has_contact * torch.ones_like(pos_err)
    
    # outgoing ball bounce
    pos_err = torch.sum((bounce_pos - target_pos) **2, dim=-1)
    # same reward for all steps
    ball_pos_reward = bounce_in \
        * torch.exp(- scales.get('bounce_pos', 0.05) * pos_err) \
        * torch.exp(- scales.get('bounce_time', 0.1) * bounce_time )

    # all rewards
    reward = w_pos * pos_reward + w_bounce * ball_pos_reward
    sub_rewards = torch.stack([pos_reward, ball_pos_reward], dim=-1)
    sub_rewards_names = 'pos_reward,ball_pos_reward'

    return reward, sub_rewards, sub_rewards_names