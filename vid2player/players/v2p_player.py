from rl_games.algos_torch import torch_ext

import learning.common_player as common_player

import torch
import os
from tqdm import tqdm


class V2PPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        super().__init__(config)
        self.task = self.env.task
        self.clip_actions = False

        dual_model_cp = config.get('dual_model_cp', None)
        if dual_model_cp is not None and not config.get('load_checkpoint', False):
            print("Loading dual v2p policy ...")
            self.load_dual_cp(dual_model_cp)

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
            
            # If model has larger obs dim
            network_obs_dim = network.actor_mlp[0].weight.data.shape[-1]
            weight_obs_dim = model_checkpoint['actor_mlp.0.weight'].shape[-1]
            if network_obs_dim > weight_obs_dim:
                network.actor_mlp[0].weight.data.zero_()
                network.actor_mlp[0].weight.data[:, :weight_obs_dim] = model_checkpoint['actor_mlp.0.weight']
                del model_checkpoint['actor_mlp.0.weight']
                network.critic_mlp[0].weight.data.zero_()
                network.critic_mlp[0].weight.data[:, :weight_obs_dim] = model_checkpoint['critic_mlp.0.weight']
                del model_checkpoint['critic_mlp.0.weight']

            network.load_state_dict(model_checkpoint, strict=False)

            load_rlgame_norm = len([key for key in missing_checkpoint_keys if 'running_obs' in key]) > 0
            if load_rlgame_norm:
                if 'running_mean_std' in checkpoint:
                    print('loading running_obs from rl game running_mean_std')
                    obs_len = checkpoint['running_mean_std']['running_mean'].shape[0]
                    network.running_obs.n = checkpoint['running_mean_std']['count'].long()
                    network.running_obs.mean[:obs_len] = checkpoint['running_mean_std']['running_mean'].float()
                    network.running_obs.var[:obs_len] = checkpoint['running_mean_std']['running_var'].float()
                    network.running_obs.std[:obs_len] = torch.sqrt(network.running_obs.var[:obs_len])
                elif 'running_obs.running_mean' in model_checkpoint:
                    print('loading running_obs from rl game in model')
                    obs_len = model_checkpoint['running_obs.running_mean'].shape[0]
                    network.running_obs.n = model_checkpoint['running_obs.count'].long()
                    network.running_obs.mean[:obs_len] = model_checkpoint['running_obs.running_mean'].float()
                    network.running_obs.var[:obs_len] = model_checkpoint['running_obs.running_var'].float()
                    network.running_obs.std[:obs_len] = torch.sqrt(network.running_obs.var[:obs_len])
    
    def restore(self, cp_name):
        if self.config.get('dual_model_cp', None) is not None:
            return
        if cp_name is not None:
            cp_path = os.path.join(self.network_path, f"{self.config['name']}_{cp_name}.pth")
            if os.path.exists(cp_path):
                checkpoint = torch_ext.load_checkpoint(cp_path)

                network_obs_dim = self.model.a2c_network.actor_mlp[0].weight.data.shape[-1]
                weight_obs_dim = checkpoint['model']['a2c_network.actor_mlp.0.weight'].shape[-1]
                if network_obs_dim != weight_obs_dim:
                    # If model has larger obs dim
                    print(f"Warning: network obs dim = {network_obs_dim} but weight obs dim = {weight_obs_dim}")
                    assert network_obs_dim > weight_obs_dim
                    self.model.a2c_network.actor_mlp[0].weight.data.zero_()
                    self.model.a2c_network.actor_mlp[0].weight.data[:, :weight_obs_dim] = checkpoint['model']['a2c_network.actor_mlp.0.weight']
                    del checkpoint['model']['a2c_network.actor_mlp.0.weight']
                    self.model.a2c_network.critic_mlp[0].weight.data.zero_()
                    self.model.a2c_network.critic_mlp[0].weight.data[:, :weight_obs_dim] = checkpoint['model']['a2c_network.critic_mlp.0.weight']
                    del checkpoint['model']['a2c_network.critic_mlp.0.weight']
                    self.model.load_state_dict(checkpoint['model'], strict=False)

                    self.running_mean_std.count = checkpoint['running_mean_std']['count']
                    self.running_mean_std.running_mean[:weight_obs_dim] = checkpoint['running_mean_std']['running_mean']
                    self.running_mean_std.running_var[:weight_obs_dim] = checkpoint['running_mean_std']['running_var']
                else:
                    self.model.load_state_dict(checkpoint['model'])
                    if self.normalize_input:
                        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            else:
                print(f'Checkpoint {cp_path} does not exist, trying pretrained checkpoint ...')
                pretrained_model_cp = self.config.get('pretrained_model_cp', None)
                if pretrained_model_cp is not None:
                    checkpoint = torch_ext.load_checkpoint(pretrained_model_cp)
                    self.model.load_state_dict(checkpoint['model'])
                    if self.normalize_input:
                        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
                else:
                    print('Pretrained checkpoint does not exist either')
        else:
            print('No checkpoint provided.')
    
    def get_action(self, obs_dict, is_determenistic=False):
        obs = obs_dict['obs']
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
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
        return current_action

    def run(self):
        is_determenistic = self.is_determenistic
        max_steps = self.config['max_test_steps']

        obs_dict = self.env_reset()
        batch_size = 1
        batch_size = self.get_batch_size(obs_dict['obs'], batch_size)
        
        done_indices = []
        self.task.render_vis(init=True)
        for n in tqdm(range(max_steps)):
            obs_dict = self.env_reset(done_indices)

            action = self.get_action(obs_dict, is_determenistic)

            obs_dict, r, done, info = self.env_step(self.env, action)
            
            self.task.render_vis()

            self._post_step(info)

            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            done_indices = done_indices[:, 0]