from rl_games.algos_torch import torch_ext
from rl_games.common.player import BasePlayer

import learning.common_player as common_player

import torch
import os
import numpy as np


class ImitatorPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.args = config['args']
        self.task = self.env.task
        self.network = config['network']
        self.network_path = config['network_path']
        
        self._setup_action_space()
        self.mask = [False]

        self.clip_actions = False

        self.normalize_input = self.config['normalize_input']
        
        net_config = self._build_net_config()
        self._build_net(net_config)

        if hasattr(self.task, 'prepare_model'):
            self.task.prepare_model(self.model)

        pretrained_model_cp = config.get('pretrained_model_cp', None)
        if self.args.checkpoint == 'base' and pretrained_model_cp is not None and not config.get('load_checkpoint', False):
            if type(pretrained_model_cp) is list:
                for cp in pretrained_model_cp:
                    self.load_pretrained(cp)
            else:
                self.load_pretrained(pretrained_model_cp)

        dual_model_cp = config.get('dual_model_cp', None)
        if self.args.checkpoint == 'base' and dual_model_cp is not None and not config.get('load_checkpoint', False):
            self.load_dual_cp(dual_model_cp)

    def restore(self, cp_name):
        if cp_name is not None and cp_name != "base":
            cp_path = os.path.join(self.network_path, f"{self.config['name']}_{cp_name}.pth")
            checkpoint = torch.load(cp_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            if self.normalize_input:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        else:
            print('No checkpoint provided.')

    def load_dual_cp(self, dual_model_cp):
        networks = [self.model.a2c_network.network1, self.model.a2c_network.network2]
        for network, cp in zip(networks, dual_model_cp):
            checkpoint = torch.load(cp, map_location=self.device)
            model_checkpoint = {x[len('a2c_network.'):]: y for x, y in checkpoint['model'].items()}

            model_state_dict = network.state_dict()
            missing_checkpoint_keys = [key for key in model_state_dict if key not in model_checkpoint]
            print(f'loading model from EmbodyPose checkpoint {cp} ...')
            print('Shared keys in model and checkpoint:', [key for key in model_state_dict if key in model_checkpoint])
            print('Keys not found in current model:', [key for key in model_checkpoint if key not in model_state_dict])
            print('Keys not found in checkpoint:', missing_checkpoint_keys)

            discard_pretrained_sigma = self.config.get('discard_pretrained_sigma', False)
            if discard_pretrained_sigma:
                checkpoint_keys = list(model_checkpoint)
                for key in checkpoint_keys:
                    if 'sigma' in key:
                        model_checkpoint.pop(key)

            network.load_state_dict(model_checkpoint, strict=False)

            load_rlgame_norm = len([key for key in missing_checkpoint_keys if 'running_obs' in key]) > 0
            if load_rlgame_norm:
                if 'running_mean_std' in checkpoint:
                    print('loading running_obs from rl game running_mean_std')
                    network.running_obs.n = checkpoint['running_mean_std']['count'].long()
                    network.running_obs.mean[:] = checkpoint['running_mean_std']['running_mean'].float()
                    network.running_obs.var[:] = checkpoint['running_mean_std']['running_var'].float()
                    network.running_obs.std[:] = torch.sqrt(network.running_obs.var)
                elif 'running_obs.running_mean' in model_checkpoint:
                    print('loading running_obs from rl game in model')
                    obs_len = model_checkpoint['running_obs.running_mean'].shape[0]
                    network.running_obs.n = model_checkpoint['running_obs.count'].long()
                    network.running_obs.mean[:obs_len] = model_checkpoint['running_obs.running_mean'].float()
                    network.running_obs.var[:obs_len] = model_checkpoint['running_obs.running_var'].float()
                    network.running_obs.std[:obs_len] = torch.sqrt(network.running_obs.var[:obs_len])

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
            if 'running_mean_std' in weights:
                self.running_mean_std.load_state_dict(weights['running_mean_std'])
            else:
                self.running_mean_std.count = weights['model']['a2c_network.running_obs.n'].long()
                self.running_mean_std.running_mean[:] = weights['model']['a2c_network.running_obs.mean'].float()
                self.running_mean_std.running_var[:] = weights['model']['a2c_network.running_obs.var'].float()

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents
        }
        if hasattr(self.task, 'num_motion_frames'):
            config['num_motion_frames'] = self.task.num_motion_frames
        if hasattr(self.task, 'smpl_rest_joints'):
            config['smpl_rest_joints'] = self.task.smpl_rest_joints
            config['smpl_parents'] = self.task.smpl_parents
            config['smpl_children'] = self.task.smpl_children
        return config

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos
        
    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        value = self.model.a2c_network.eval_critic(obs_dict)
        return value

    def run_one_step(self):
        # get observation
        obs = torch.clamp(self.task.obs_buf, -self.env.clip_obs, self.env.clip_obs).to(self.device)
        obs_dict = self.obs_to_torch(obs)
        batch_size = 1
        batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

        # get action
        action = self.get_action(obs_dict, is_determenistic=True)
        action = torch.clamp(action, -self.env.clip_actions, self.env.clip_actions).to(self.device)

        # run one step
        self.task.step(action)

        if not self.task.headless:
            self.task.render_vis(init=False)
    
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

            for n in range(self.max_steps):
                
                obs_dict = self.env_reset(done_indices)
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

                all_done_indices = done.nonzero(as_tuple=False)
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

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return