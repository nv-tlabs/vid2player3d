import torch
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ImitatorModel(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('im', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ImitatorModel.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states, extra = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'extra' : extra
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma,
                    'extra' : extra
                }
                return result