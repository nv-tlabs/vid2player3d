from rl_games.algos_torch import network_builder

from .im_network_builder import ImitatorBuilder

import torch
import torch.nn as nn


class ImitatorBuilderDual(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            self.network1 = ImitatorBuilder.Network(params, **kwargs)
            self.network2 = ImitatorBuilder.Network(params, **kwargs)

        def load(self, params):
            self.params = params
            self.network1.load(params)
            self.network2.load(params)
            return

        def is_rnn(self):
            return False

        def forward(self, obs_dict):
            actor_outputs = self.eval_actor(obs_dict)
            value = self.eval_critic(obs_dict)

            output = actor_outputs + (value, None, dict())

            return output

        def eval_actor(self, obs_dict):
            obs_dict1 = obs_dict.copy()
            obs_dict1['obs'] = obs_dict1['obs'][0::2]
            obs_dict2 = obs_dict.copy()
            obs_dict2['obs'] = obs_dict2['obs'][1::2]
            actor_outputs1 = self.network1.eval_actor(obs_dict1)
            actor_outputs2 = self.network2.eval_actor(obs_dict2)
            actor_outputs = tuple(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).reshape(-1, *x.shape[1:]) for x, y in zip(actor_outputs1, actor_outputs2))
            return actor_outputs

        def eval_critic(self, obs_dict):
            obs_dict1 = obs_dict.copy()
            obs_dict1['obs'] = obs_dict1['obs'][0::2]
            obs_dict2 = obs_dict.copy()
            obs_dict2['obs'] = obs_dict2['obs'][1::2]
            value1 = self.network1.eval_critic(obs_dict1)
            value2 = self.network2.eval_critic(obs_dict2)
            value = torch.cat([value1.unsqueeze(1), value2.unsqueeze(1)], dim=1).reshape(-1, *value1.shape[1:])
            return value

    def build(self, name, **kwargs):
        net = ImitatorBuilderDual.Network(self.params, **kwargs)
        return net