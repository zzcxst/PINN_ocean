import torch
from torch import nn
import numpy as np

w = 30
s = 100

## SIREN 
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(w * input)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w, np.sqrt(6 / num_input) / w)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            #m.weight.uniform_(-np.sqrt(9 / num_input) / w, np.sqrt(9 / num_input) / w)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

## PINN
class PINN(nn.Module):
    def __init__(self, inputs, outputs, hiddens, n_layers, defult='siren'):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(inputs, hiddens))
            if defult == 'siren':
                layers.append(Sine())
            elif defult == 'relu':
                layers.append(nn.ReLU())
            elif defult == 'tanh':
                layers.append(nn.Tanh())
            else:
                print('error: Activation function not configured')
            inputs = hiddens

        self.net = nn.Sequential(
            *layers,
            nn.Linear(hiddens, outputs)
        )
        
    def forward(self, r):
        return self.net(r)
        
## PINN-Multiscale
class PINN_MS(nn.Module):
    def __init__(self, inputs, outputs, hiddens, n_layers, n_scales):
        super().__init__()
        layers = []
        self.scale_branches = nn.ModuleList()
        for _ in range(n_layers):
            layers.append(nn.Linear(inputs, hiddens))
            layers.append(Sine())
            inputs = hiddens
        for _ in range(n_scales):
            branch = nn.Sequential(*layers)
            self.scale_branches.append(branch)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hiddens * n_scales, hiddens),
            Sine(),
            nn.Linear(hiddens, outputs)
            )
    def forward(self, r):
        branch_outputs = []
        i = 0
        scale_factor = [1, 0.5, 2]
        for branch in self.scale_branches:
            branch_output = branch(r * scale_factor[i])
            branch_outputs.append(branch_output)
            i = i + 1

        fused_features = torch.cat(branch_outputs, dim=1)

        output = self.fusion_layer(fused_features)
        return output
