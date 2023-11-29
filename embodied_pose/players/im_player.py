import torch
import os
import numpy as np
import time

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import rescale_actions, unsqueeze_obs
from rl_games.common.player import BasePlayer

import learning.common_player as common_player


class ImitatorPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.args = config['args']
        self.task = self.env.task
        self.horizon_length = config['horizon_length']
        self.num_actors = config['num_actors']
        self.network = config['network']
        self.network_path = config['network_path']
        
        self._setup_action_space()
        self.mask = [False]

        self.clip_actions = False

        self.normalize_input = self.config['normalize_input']
        
        net_config = self._build_net_config()
        self._build_net(net_config)  

        self.task.register_model(self.model)

        pretrained_model_cp = config.get('pretrained_model_cp', None)
        if self.args.checkpoint == 'base' and pretrained_model_cp is not None and not config.get('load_checkpoint', False):
            if type(pretrained_model_cp) is list:
                for cp in pretrained_model_cp:
                    self.load_pretrained(cp)
            else:
                self.load_pretrained(pretrained_model_cp)

    def restore(self, cp_name):
        if cp_name is not None and cp_name != "base":
            cp_path = os.path.join(self.network_path, f"{self.config['name']}_{cp_name}.pth")
            checkpoint = torch.load(cp_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            if self.normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        else:
            print('No checkpoint provided.')

    def load_pretrained(self, cp_path):
        checkpoint = torch.load(cp_path, map_location=self.device)

        if 'pytorch-lightning_version' in checkpoint:
            model_state_dict = self.model.a2c_network.state_dict()
            print('loading model from Lighting checkpoint...')
            print('Shared keys in model and lightning checkpoint:', [key for key in checkpoint['state_dict'] if key in model_state_dict])
            print('Lightning keys not found in current model:', [key for key in checkpoint['state_dict'] if key not in model_state_dict])
            print('Keys not found in lightning checkpoint:', [key for key in model_state_dict if key not in checkpoint['state_dict']])
            self.model.a2c_network.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model_state_dict = self.model.state_dict()
            missing_checkpoint_keys = [key for key in model_state_dict if key not in checkpoint['model']]
            print('loading model from EmbodyPose checkpoint...')
            print('Shared keys in model and checkpoint:', [key for key in model_state_dict if key in checkpoint['model']])
            print('Keys not found in current model:', [key for key in checkpoint['model'] if key not in model_state_dict])
            print('Keys not found in checkpoint:', missing_checkpoint_keys)

            discard_pretrained_sigma = self.config.get('discard_pretrained_sigma', False)
            if discard_pretrained_sigma:
                checkpoint_keys = list(checkpoint['model'])
                for key in checkpoint_keys:
                    if 'sigma' in key:
                        checkpoint['model'].pop(key)

            self.set_weights(checkpoint)

            load_rlgame_norm = len([key for key in missing_checkpoint_keys if 'running_obs' in key]) > 0
            if load_rlgame_norm:
                if 'running_mean_std' in checkpoint:
                    print('loading running_obs from rl game running_mean_std')
                    self.model.a2c_network.running_obs.n = checkpoint['running_mean_std']['count'].long()
                    self.model.a2c_network.running_obs.mean[:] = checkpoint['running_mean_std']['running_mean'].float()
                    self.model.a2c_network.running_obs.var[:] = checkpoint['running_mean_std']['running_var'].float()
                    self.model.a2c_network.running_obs.std[:] = torch.sqrt(self.model.a2c_network.running_obs.var)
                elif 'a2c_network.running_obs.running_mean' in checkpoint['model']:
                    print('loading running_obs from rl game in model')
                    obs_len = checkpoint['model']['a2c_network.running_obs.running_mean'].shape[0]
                    self.model.a2c_network.running_obs.n = checkpoint['model']['a2c_network.running_obs.count'].long()
                    self.model.a2c_network.running_obs.mean[:obs_len] = checkpoint['model']['a2c_network.running_obs.running_mean'].float()
                    self.model.a2c_network.running_obs.var[:obs_len] = checkpoint['model']['a2c_network.running_obs.running_var'].float()
                    self.model.a2c_network.running_obs.std[:obs_len] = torch.sqrt(self.model.a2c_network.running_obs.var[:obs_len])

    def set_weights(self, weights):
        if hasattr(self.model, 'load_weights'):
            self.model.load_weights(weights['model'])
        else:
            self.model.load_state_dict(weights['model'], strict=False)
        
        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors,
            'smpl_rest_joints': self.task.smpl_rest_joints,
            'smpl_parents': self.task.smpl_parents,
            'smpl_children': self.task.smpl_children
        }
        if hasattr(self.task, 'num_con_planes'):
            config['num_con_planes'] = self.task.num_con_planes
            config['num_con_bodies'] = self.task.num_con_bodies
        return config

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            't': obs['t']
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def get_action(self, obs_dict, is_determenistic = False):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            't': obs_dict['t'],
            'global_t_offset': obs_dict['global_t_offset'],
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            obs_dict = {'obs': obs}
            return obs_dict, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            self.task.render_vis(init=True)

            prev_dones = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            for n in range(self.max_steps):
                t = n % self.task.context_length
                if n > 0 and t == 0:
                    self.task._init_context(self.task._reset_ref_motion_ids, self.task._cur_ref_motion_times)
                
                obs_dict['t'] = t
                obs_dict['global_t_offset'] = n - t
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)

                obs_dict, r, done, info =  self.env_step(self.env, action)

                cr += r
                steps += 1

                self.task.render_vis()
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                step_dones = done * (1 - prev_dones)
                all_done_indices = step_dones.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
                
                done_indices = done_indices[:, 0]

                prev_dones = done.clone()
                if done[0]:
                    break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return