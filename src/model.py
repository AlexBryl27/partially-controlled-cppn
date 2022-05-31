import torch
import torch.nn as nn


class CPPN(nn.Module):
    

    def __init__(self, inp_dim=3, hid_dim=32, n_layers=8):
        super(CPPN, self).__init__()

        self.n_layers = n_layers

        self.input_layer = nn.Linear(inp_dim, hid_dim)
        self.layer = nn.Linear(hid_dim, hid_dim)
        self.out_layer = nn.Linear(hid_dim, 3)
        

    def forward(self, inputs):
        x = self.input_layer(inputs)
        x = torch.tanh(x)
        x = torch.clip(x, -1., 1.)
        for i in range(self.n_layers):
            x = self.layer(x)
            x = torch.tanh(x)
            x = torch.clip(x, -1., 1.)
        x = self.out_layer(x)
        x = torch.tanh(x)
        x = torch.clip(x, -1., 1.)
        return x
