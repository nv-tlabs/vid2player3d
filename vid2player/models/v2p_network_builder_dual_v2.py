from rl_games.algos_torch import network_builder

from .v2p_network_builder import V2PBuilder

import torch
import torch.nn as nn


class V2PBuilderDualV2(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            super().__init__()
            self.network1 = V2PBuilder.Network(params, **kwargs)
            self.network2 = V2PBuilder.Network(params, **kwargs)

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
            obs_dict1['obs'] = obs_dict1['obs'][obs_dict['env_ids_1']]
            obs_dict2 = obs_dict.copy()
            obs_dict2['obs'] = obs_dict2['obs'][obs_dict['env_ids_2']]
            if obs_dict1['obs'].shape[0] > 0:
                actor_outputs1 = self.network1.eval_actor(obs_dict1)
            if obs_dict2['obs'].shape[0] > 0:
                actor_outputs2 = self.network2.eval_actor(obs_dict2)
            if obs_dict1['obs'].shape[0] == 0:
                return actor_outputs2
            elif obs_dict2['obs'].shape[0] == 0:
                return actor_outputs1
            actor_outputs = []
            for x, y in zip(actor_outputs1, actor_outputs2):
                xy = torch.cat([x, y], dim=0)
                xy[obs_dict['env_ids_1']] = x
                xy[obs_dict['env_ids_2']] = y
                actor_outputs += [xy]
            return tuple(actor_outputs)

        def eval_critic(self, obs_dict):
            obs_dict1 = obs_dict.copy()
            obs_dict1['obs'] = obs_dict1['obs'][obs_dict['env_ids_1']]
            obs_dict2 = obs_dict.copy()
            obs_dict2['obs'] = obs_dict2['obs'][obs_dict['env_ids_2']]
            if obs_dict1['obs'].shape[0] > 0:
                value1 = self.network1.eval_critic(obs_dict1)
            if obs_dict2['obs'].shape[0] > 0:
                value2 = self.network2.eval_critic(obs_dict2)
            if obs_dict1['obs'].shape[0] == 0:
                return value2
            elif obs_dict2['obs'].shape[0] == 0:
                return value1
            value = torch.cat([value1, value2], dim=0)
            value[obs_dict['env_ids_1']] = value1
            value[obs_dict['env_ids_2']] = value2
            return value

    def build(self, name, **kwargs):
        net = V2PBuilderDualV2.Network(self.params, **kwargs)
        return net