import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import msgpack
import os
from .utils import msgpack_numpy_patch
msgpack_numpy_patch()


class Network(nn.Module):
    def __init__(self, device, lr,  input_dim, output_dim):
        super(Network, self).__init__()

        self.device = device
        self.lr = lr

        self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
        nn.Linear(input_dim, 32),
        self.activation,
        nn.Linear(32, 64),
        self.activation,
        nn.Linear(64, 64),
        self.activation,
        nn.Linear(64, 32),
        self.activation,
        nn.Linear(32, output_dim),  
    ) 
        
        self.optimizer = optim.Adam(self.parameters(), self.lr)

        self.loss = nn.SmoothL1Loss(reduction='mean')

        self.to(self.device)
        
    def forward(self, input):
        return self.net(input)
    
    def action(self, obs):
        """
        Select the action with the max q value according to the model

        Args:
            obs: The current observation

        Returns:
            action (int): The action (an index) with the max q value
        """
        obs_t = torch.from_numpy(obs).to(self.device)
        q_values = self(obs_t)
        max_q_index = torch.argmax(q_values)
        action = max_q_index.detach().item()
        return action
    
    def save(self, save_path, step, episode_count, rew_mean, len_mean, suc_mean, fail_mean, col_mean ):
        """
        Save model's weights and other informations about training

        Args:
            save_path: The path where to save the weights
            step: The number of steps
            episode_count: The number of episodes
            rew_mean: The mean reward over the last k episodes 
            len_mean: The mean length over the last k episodes
        """
        params_dict = {
            'parameters': {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()},
            'step': step, 'episode_count': episode_count, 'rew_mean': rew_mean, 'len_mean': len_mean, 'suc_mean': suc_mean, 'fail_mean' : fail_mean, 'col_mean' : col_mean
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(msgpack.dumps(params_dict))

    def load(self, load_path):
        """
        Load model's weights and other informations about training

        Args:
            load_path: The path from where to load the model

        Returns:
            step: The number of steps
            episode_count: The number of episodes
            rew_mean: The mean reward over the last k episodes 
            len_mean: The mean length over the last k episodes
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_dict = msgpack.loads(f.read())

        parameters = {k: torch.as_tensor(np.array(v), device=self.device) for k, v in params_dict['parameters'].items()}
        self.load_state_dict(parameters)

        return params_dict['step'], params_dict['episode_count'], params_dict['rew_mean'], params_dict['len_mean'], params_dict['suc_mean'], params_dict['fail_mean'], params_dict['col_mean']
    
    def load_test(self, load_path):
        """
        Load model's weights and other informations about training

        Args:
            load_path: The path from where to load the model

        Returns:
            step: The number of steps
            episode_count: The number of episodes
            rew_mean: The mean reward over the last k episodes 
            len_mean: The mean length over the last k episodes
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_dict = msgpack.loads(f.read())
            

        parameters = {k: torch.as_tensor(np.array(v), device=self.device) for k, v in params_dict['parameters'].items()}
        self.load_state_dict(parameters)

        return 
    

class DuelingDeepQNetwork(Network):
    def __init__(self, device, lr, input_dim, output_dim):
        super(DuelingDeepQNetwork, self).__init__(device, lr,  input_dim, output_dim)

        self.fc_val = nn.Linear(output_dim, 1)
        self.fc_adv = nn.Linear(output_dim, output_dim)
        # self.aggregate_layer = (lambda val, adv: torch.add(val, (adv - adv.mean(dim=1, keepdim=True))))
        self.aggregate_layer = (lambda val, adv: torch.add(val, (adv - adv.mean())))

        self.to(self.device)

    def forward(self, s):
        net = self.net(s)
        val = self.fc_val(net)
        adv = self.fc_adv(net)
        agg = self.aggregate_layer(val, adv)

        return agg

    def value(self, s):
        net = self.net(s)
        val = self.fc_val(net)

        return val

    def advantages(self, s):
        net = self.net(s)
        adv = self.fc_adv(net)

        return adv

    def actions(self, obses):
        obses_t = torch.as_tensor(obses, dtype=torch.float32).to(self.device)
        adv_q_values = self.advantages(obses_t)

        max_adv_q_indices = torch.argmax(adv_q_values, dim=1)
        actions = max_adv_q_indices.detach().tolist()

        return actions
