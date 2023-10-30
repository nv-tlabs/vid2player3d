from env.tasks.physics_mvae_controller import PhysicsMVAEController
from utils.common import get_opponent_env_ids

import torch


class PhysicsMVAEControllerDual(PhysicsMVAEController):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params, 
                         physics_engine=physics_engine, 
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._reset_reaction_buf[:] = 0
        self._reset_recovery_buf[:] = 0

    def create_sim(self):
        super().create_sim()
    
    def reset(self, env_ids=None):
        if env_ids is None:
            if self._has_init: return
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
    
    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            assert len(env_ids) % 2 == 0, len(env_ids)
            if self.cfg_v2p.get('serve_from', 'near') == 'near':
                reset_actor_reaction_env_ids = env_ids[::2]
            else:
                reset_actor_reaction_env_ids = env_ids[1::2]

            reset_actor_recovery_env_ids = get_opponent_env_ids(reset_actor_reaction_env_ids)
            self._reset_reaction_buf[reset_actor_reaction_env_ids] = 1
            self._reset_recovery_buf[reset_actor_recovery_env_ids] = 1
        else:
            reset_actor_reaction_env_ids = reset_actor_recovery_env_ids = torch.LongTensor([])
        
        reset_reaction_env_ids = self._reset_reaction_buf.nonzero(as_tuple=False).flatten()
        reset_recovery_env_ids = self._reset_recovery_buf.nonzero(as_tuple=False).flatten()
        reset_actor_env_ids = env_ids

        if len(env_ids) > 0:
            self._mvae_player.reset_dual(reset_actor_reaction_env_ids, reset_actor_recovery_env_ids)
            self._reset_env_tensors(reset_actor_env_ids)
        
        if len(reset_reaction_env_ids) > 0:
            new_traj = self._physics_player.task.reset(
                reset_actor_reaction_env_ids, reset_reaction_env_ids)
            self._ball_traj[reset_reaction_env_ids, :new_traj.shape[1]] = new_traj.to(self.device)
            
            if not self.headless and not self._has_init:
                self._physics_player.task.render_vis(init=True)

            self._update_state()
        
        if len(reset_recovery_env_ids) > 0:
            self._reset_recovery_tasks(reset_recovery_env_ids)
        if len(reset_reaction_env_ids) > 0:
            self._reset_reaction_tasks(reset_reaction_env_ids, reset_actor_reaction_env_ids)
            self._compute_observations(reset_reaction_env_ids)
        
        self._has_init = True

    def _reset_reaction_tasks(self, env_ids, humanoid_env_ids=None):
        if self.cfg_v2p.get('use_history_ball_obs'):
            self._ball_obs[env_ids] = self._ball_pos[env_ids].view(-1, 1, 3).repeat(1, self._obs_ball_traj_length, 1)
        self._tar_time[env_ids] = 0
        self._tar_action[env_ids] = 1
        self._num_reset_reaction[env_ids] += 1
        self._bounce_in[env_ids] = 0
    
        if self.cfg_v2p.get('use_random_ball_target'):
            rand_seed = torch.rand((len(env_ids)))
            bounce_left = rand_seed < 0.33
            bounce_right = rand_seed > 0.67
            bounce_middle = ~bounce_left & ~bounce_right
            self._target_bounce_pos[env_ids[bounce_left]] = torch.FloatTensor([-3, 10, 0]).to(self.device)
            self._target_bounce_pos[env_ids[bounce_middle]] = torch.FloatTensor([0, 10, 0]).to(self.device)
            self._target_bounce_pos[env_ids[bounce_right]] = torch.FloatTensor([3, 10, 0]).to(self.device)

        if self.cfg_v2p['reward_type'] == 'return_w_estimate':
            self._est_bounce_pos[env_ids, :] = 0
            self._est_bounce_time[env_ids] = 0
            self._est_bounce_in[env_ids] = 0
            self._est_max_height[env_ids] = 0
        
    def _reset_recovery_tasks(self, env_ids):
        self._tar_action[env_ids] = 0
        self._physics_player.task._has_bounce[env_ids] = 0
        self._physics_player.task._bounce_pos[env_ids] = 0
    
    def _compute_reset(self):
        has_contact = self._physics_player.task._has_racket_ball_contact
        has_bounce = self._physics_player.task._has_bounce
        in_recovery = self._tar_action == 0

        miss_ball = self._ball_pos[:, 1] < self._root_pos[:, 1] - 1
        ball_bounce_twice = has_bounce & (self._ball_pos[:, 2] < 0.05)
        self._reset_recovery_buf = (self._tar_action == 1) & \
            (has_contact | miss_ball | ball_bounce_twice)
        
        self._compute_stats()

        # recovery also marks the reaction of their opponent
        self._reset_reaction_buf[::2] = self._reset_recovery_buf[1::2]
        self._reset_reaction_buf[1::2] = self._reset_recovery_buf[::2]

        miss = self._reset_recovery_buf & ~has_contact
        out = in_recovery & has_bounce & ~self._bounce_in
        terminate = miss | out
        if terminate.sum() > 0:
            terminate[::2] |= terminate[1::2]
            terminate[1::2] |= terminate[::2]

            self.reset_buf[terminate] = 1
            self._reset_reaction_buf[terminate] = 0
            self._reset_recovery_buf[terminate] = 0