from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

import learning.common_agent as common_agent 
from utils.common import AverageMeter, get_eta_str

import torch 
import time
import os
import numpy as np
from torch import nn


class V2PAgent(common_agent.CommonAgent):

    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.sub_rewards_names = None
        self.log_dict = dict()
        self.task = self.vec_env.env.task

    def restore(self, cp_name):
        cp_path = os.path.join(self.network_path, f"{self.config['name']}_{cp_name}.pth")
        checkpoint = torch_ext.load_checkpoint(cp_path)
        self.set_full_state_weights(checkpoint)
    
    def load_pretrained(self, cp_path):
        checkpoint = torch_ext.load_checkpoint(cp_path)
        model_state_dict = self.model.state_dict()

        missing_checkpoint_keys = [key for key in model_state_dict if key not in checkpoint['model']]
        print('Keys not found in current model:', [key for key in checkpoint['model'] if key not in model_state_dict])
        print('Keys not found in checkpoint:', missing_checkpoint_keys)

        discard_pretrained_sigma = self.config.get('discard_pretrained_sigma', False)
        if discard_pretrained_sigma:
            print('Discard sigma from pretrained model')
            checkpoint_keys = list(checkpoint['model'])
            for key in checkpoint_keys:
                if 'sigma' in key:
                    checkpoint['model'].pop(key)
        
        self.set_network_weights(checkpoint['model'])
        self.set_stats_weights(checkpoint)

    def set_network_weights(self, weights):
        network_obs_dim = self.model.a2c_network.actor_mlp[0].weight.data.shape[-1]
        weight_obs_dim = weights['a2c_network.actor_mlp.0.weight'].shape[-1]

        network_action_dim = self.model.a2c_network.mu.weight.data.shape[0]
        weight_action_dim = weights['a2c_network.mu.weight'].shape[0]

        if network_obs_dim != weight_obs_dim:
            print(f"Warning: network obs dim = {network_obs_dim} but weight obs dim = {weight_obs_dim}")
            assert network_obs_dim > weight_obs_dim
            self.model.a2c_network.actor_mlp[0].weight.data.zero_()
            self.model.a2c_network.actor_mlp[0].weight.data[:, :weight_obs_dim] = weights['a2c_network.actor_mlp.0.weight']
            del weights['a2c_network.actor_mlp.0.weight']

            self.model.a2c_network.critic_mlp[0].weight.data.zero_()
            self.model.a2c_network.critic_mlp[0].weight.data[:, :weight_obs_dim] = weights['a2c_network.critic_mlp.0.weight']
            del weights['a2c_network.critic_mlp.0.weight']
        
        if network_action_dim != weight_action_dim:
            print(f"Warning: network action dim = {network_action_dim} but weight action dim = {weight_action_dim}")
            assert network_action_dim > weight_action_dim
            self.model.a2c_network.mu.weight.data.zero_()
            self.model.a2c_network.mu.weight.data[:weight_action_dim, :] = weights['a2c_network.mu.weight']
            self.model.a2c_network.mu.bias.data.zero_()
            self.model.a2c_network.mu.bias.data[:weight_action_dim] = weights['a2c_network.mu.bias']
            del weights['a2c_network.mu.weight']
            del weights['a2c_network.mu.bias']
            if 'a2c_network.sigma' in weights:
                del weights['a2c_network.sigma']

        self.model.load_state_dict(weights, strict=False)

    def set_stats_weights(self, weights):
        if self.normalize_input:
            assert 'running_mean_std' in weights
            network_obs_dim = self.running_mean_std.running_mean.shape[0]
            weight_obs_dim = weights['running_mean_std']['running_mean'].shape[0]
            if network_obs_dim != weight_obs_dim:
                assert network_obs_dim > weight_obs_dim
                self.running_mean_std.count = weights['running_mean_std']['count']
                self.running_mean_std.running_mean[:weight_obs_dim] = weights['running_mean_std']['running_mean']
                self.running_mean_std.running_var[:weight_obs_dim] = weights['running_mean_std']['running_var']
            else:
                self.running_mean_std.load_state_dict(weights['running_mean_std'])
         
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.has_central_value:
            self.central_value_net.set_stats_weights(weights['assymetric_vf_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def train(self):
        args = self.config['args']
        self.init_tensors()
        self.last_mean_rewards = -100500
        self.best_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        pretrained_model_cp = self.config.get('pretrained_model_cp', None)
        if pretrained_model_cp is not None and not self.config.get('load_checkpoint', False):
            self.load_pretrained(pretrained_model_cp)

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        self._init_train()

        while True:
            if hasattr(self.task, 'pre_epoch'):
                self.task.pre_epoch(self.epoch_num)

            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            play_time = train_info['play_time']
            update_time = train_info['update_time']
            total_time += sum_time
            frame = self.frame
            if self.multi_gpu:
                self.hvd.sync_stats(self)

            # logging
            if self.rank == 0:
                self.log_dict = dict()
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames

                aux_losses_dict = {k: torch_ext.mean_list(v).item() for k, v in train_info.items() if k.startswith('aux_')}
                if self.print_stats:
                    fps_step = curr_frames // scaled_play_time
                    fps_total = curr_frames // scaled_time
                    mean_lengths = self.game_lengths.get_mean() if self.game_rewards.current_size > 0 else 0
                    print('{}\tT_play {:.2f}\tT_update {:.2f}\tETA {}\tstep_rewards {:.4f} {}\teps_len {:.2f}\tfps step {}\tfps total {}\t{}'
                          .format(epoch_num, play_time, update_time, get_eta_str(epoch_num, self.max_epochs, sum_time), self.step_rewards.avg.item(),
                                    np.array2string(self.step_sub_rewards.avg.cpu().numpy(), formatter={'all': lambda x: '%.4f' % x}, separator=',') \
                                        if self.sub_rewards_names is not None else '',
                                    mean_lengths, fps_step, fps_total, self.config['args'].cfg))

                self.log_dict.update({'frame': frame, 'epoch_num': epoch_num, 'total_time': total_time})
                self.log_dict.update({'performance/total_fps': curr_frames / scaled_time})
                self.log_dict.update({'performance/step_fps': curr_frames / scaled_play_time})
                self.log_dict.update({'performance/update_time': train_info['update_time']})
                self.log_dict.update({'performance/play_time': train_info['play_time']})
                self.log_dict.update({'losses/a_loss': torch_ext.mean_list(train_info['actor_loss']).item()})
                self.log_dict.update({'losses/c_loss': torch_ext.mean_list(train_info['critic_loss']).item()})
                self.log_dict.update({'losses/bounds_loss': torch_ext.mean_list(train_info['b_loss']).item()})
                self.log_dict.update({'losses/entropy': torch_ext.mean_list(train_info['entropy']).item()})
                for key, value in aux_losses_dict.items():
                    self.log_dict['aux_losses/' + key] = value
                
                self.log_dict.update({'info/epochs': epoch_num})
                self.log_dict.update({'info/last_lr': train_info['last_lr'][-1] * train_info['lr_mul'][-1]})
                self.log_dict.update({'info/lr_mul': train_info['lr_mul'][-1]})
                self.log_dict.update({'info/e_clip': self.e_clip * train_info['lr_mul'][-1]})
                self.log_dict.update({'info/clip_frac': torch_ext.mean_list(train_info['actor_clip_frac']).item()})
                self.log_dict.update({'info/kl': torch_ext.mean_list(train_info['kl']).item()})

                self.log_dict.update({'step_rewards': self.step_rewards.avg.item()})
                if self.sub_rewards_names is not None:
                    for i, name in enumerate(self.sub_rewards_names.split(',')):
                        self.log_dict.update({f'sub_rewards/{name}': self.step_sub_rewards.avg[i].item()})

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.log_dict.update({'game_rewards{0}'.format(i): mean_rewards[i]})
                    self.log_dict.update({'episode_lengths': mean_lengths})

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)
                
                # save models
                latest_model_file = os.path.join(self.network_path, f"{self.config['name']}_latest")
                best_model_file = os.path.join(self.network_path, f"{self.config['name']}_best")
                if self.save_best_after > 0:
                    if (epoch_num % self.save_best_after == 0):
                        self.save(latest_model_file)
                if self.save_freq > 0:
                    if (epoch_num % self.save_freq == 0):
                        epoch_model_file = os.path.join(self.network_path, f"{self.config['name']}_epoch{epoch_num:05d}")
                        self.save(epoch_model_file)
                if self.game_rewards.current_size > 0 and mean_rewards[0] > self.best_mean_rewards + 1:
                    self.best_mean_rewards = mean_rewards[0]
                    print("Update best mean game rewards => ", self.best_mean_rewards)
                    self.save(best_model_file)
                if epoch_num >= self.max_epochs:
                    self.save(latest_model_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num

                update_time = 0
        return
    
    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        value = self.model.a2c_network.eval_critic(obs_dict)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def play_steps(self):
        self.set_eval()
        
        done_indices = []
        update_list = self.update_list

        self.step_rewards = AverageMeter()
        self.step_sub_rewards = AverageMeter()

        self.obs = self.env_reset(done_indices)
        for n in range(self.horizon_length):
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic({'obs': self.obs['obs']})
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.step_rewards.update(rewards.mean(dim=0))
            if infos['sub_rewards'] is not None:
                self.step_sub_rewards.update(infos['sub_rewards'].mean(dim=0))
                self.sub_rewards_names = infos['sub_rewards_names']
            # NOTE: only save the last step's info for simplicity
            self.infos = infos

            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

            self.obs = self.env_reset(done_indices)

            # For debug
            self.task.render_vis()

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss

            if hasattr(self.task, 'get_aux_losses'):
                aux_losses, aux_losses_weighted = self.task.get_aux_losses(res_dict)
                for aux_loss in aux_losses_weighted.values():
                    loss += aux_loss
            else:
                aux_losses = aux_losses_weighted = None
            
            a_clip_frac = torch.mean(a_info['actor_clipped'].float())
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()

        # parameters = [p for p in self.model.parameters() if p.grad is not None]
        # device = parameters[0].grad.device
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
        # print('total_norm:', total_norm)

        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        if aux_losses is not None:
            self.train_result.update(aux_losses)

        return