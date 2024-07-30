import torch

import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, memory_size, hidden_size, n_actions, num_hidden_layers = 1, device='cpu', deterministic_policy = False, dtype=torch.float32) -> None:
        assert num_hidden_layers >= 1
        super().__init__()
        network = [nn.Linear(memory_size, hidden_size, device=device, dtype=dtype), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers):
            network.append(nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype))
            network.append(nn.ReLU(inplace=True))
        network.append(nn.Linear(hidden_size, n_actions, device=device, dtype=dtype))
        if deterministic_policy:
            network.append(nn.Softmax())
        self.net = nn.Sequential(*network)
        
    def forward(self, x):
        return self.net(x)