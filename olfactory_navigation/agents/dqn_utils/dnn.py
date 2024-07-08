import torch

import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, d, h, num_hidden_layers = 1, n_actions, device='cpu', dtype=torch.float32) -> None:
        assert num_hidden_layers >= 1
        super().__init__()
        network = [nn.Linear(d, h, device=device, dtype=dtype), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers):
            network.append(nn.Linear(h, h, device=device, dtype=dtype))
            network.append(nn.ReLU(inplace=True))
        network.append(nn.Linear(h, n_actions, device=device, dtype=dtype))
        network.append(nn.Softmax())
        self.net = nn.Sequential(*network)
        
    def forward(self, x):
        return self.net(x)