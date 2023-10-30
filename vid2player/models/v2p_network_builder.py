from rl_games.algos_torch import network_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from .running_norm import RunningNorm

import torch
import torch.nn as nn

DISC_LOGIT_INIT_SCALE = 1.0


class V2PBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            self.use_running_obs = params.get('use_running_obs', False)
            self.running_obs_type = params.get('running_obs_type', 'ours')
            if self.use_running_obs:
                self.running_obs = RunningMeanStd(kwargs['input_shape'])
                if self.running_obs_type == 'rl_game':
                    self.running_obs = RunningMeanStd(kwargs['input_shape'])
                else:
                    self.running_obs = RunningNorm(kwargs['input_shape'][0])

            if self.is_continuous:
                backprop_sigma = self.space_config.get('backprop_sigma', False)
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                self.sigma = nn.Parameter(torch.zeros(kwargs.get('actions_num')), requires_grad=backprop_sigma)
                sigma_init(self.sigma)
            return

        def load(self, params):
            super().load(params)
            return

        def forward(self, obs_dict):
            if self.use_running_obs:
                obs_dict['obs_norm'] = self.running_obs(obs_dict['obs'])

            actor_outputs = self.eval_actor(obs_dict)
            value = self.eval_critic(obs_dict)

            output = actor_outputs + (value, None, dict())

            return output

        def eval_actor(self, obs_dict):
            if self.use_running_obs:
                if 'obs_norm' in obs_dict:
                    obs = obs_dict['obs_norm']
                else:
                    obs = obs_dict['obs_norm'] = self.running_obs(obs_dict['obs'])
            else:
                obs = obs_dict['obs']

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

                return mu, sigma
            return

        def eval_critic(self, obs_dict):
            if self.use_running_obs:
                if 'obs_norm' in obs_dict:
                    obs = obs_dict['obs_norm']
                else:
                    obs = obs_dict['obs_norm'] = self.running_obs(obs_dict['obs'])
            else:
                obs = obs_dict['obs']
            
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = V2PBuilder.Network(self.params, **kwargs)
        return net