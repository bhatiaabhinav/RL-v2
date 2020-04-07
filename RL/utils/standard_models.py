import torch
from torch import nn

from RL.utils.util_fns import conv2d_shape_out


class FFModel(nn.Module):
    def __init__(self, input_shape, convs, linears, deconvs=None):
        super().__init__()
        self.input_shape = input_shape

        self.num_layers = 0
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        shape = (1, input_shape[0], input_shape[1]) if len(input_shape) == 2 else input_shape
        if len(input_shape) > 1:
            for channels, kernel, stride in convs:
                conv_layer = nn.Conv2d(shape[0], channels, kernel, stride)
                shape = conv2d_shape_out(shape, channels, kernel, stride)
                self.conv_layers.append(conv_layer)
                self.num_layers += 1
        num_features = shape[0] if len(shape) == 1 else shape[0] * shape[1] * shape[2]
        for h in linears:
            hidden_layer = nn.Linear(num_features, h)
            num_features = h
            self.linear_layers.append(hidden_layer)
            self.num_layers += 1
        if deconvs:
            self.deconv_layers = nn.ModuleList()

    def forward(self, x: torch.Tensor):
        '''should be a float tensor'''
        assert 2 <= len(x.shape) <= 4
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        layers_remaining = self.num_layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            layers_remaining -= 1
            if layers_remaining > 0:
                x = torch.relu(x)
        if len(self.conv_layers) > 0:
            x = x.view(x.size(0), -1)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            layers_remaining -= 1
            if layers_remaining > 0:
                x = torch.relu(x)
        return x
