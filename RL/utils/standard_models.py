import torch
from torch import nn

from RL.utils.util_fns import conv2d_shape_out, convT2d_shape_out


class FFModel(nn.Module):
    def __init__(self, input_shape, convs, linears, unflatten_shape=None, deconvs=None):
        super().__init__()
        self.input_shape = input_shape

        self.num_layers = 0
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        shape = (1, input_shape[0], input_shape[1]) if len(
            input_shape) == 2 else input_shape
        self.convs_output_shape = None
        self.flatten_output_shape = None
        self.linears_output_shape = None
        self.unflatten_output_shape = None
        self.deconvs_output_shape = None

        if len(input_shape) > 1:
            for channels, kernel, stride in convs:
                conv_layer = nn.Conv2d(shape[0], channels, kernel, stride)
                shape = conv2d_shape_out(shape, channels, kernel, stride)
                self.conv_layers.append(conv_layer)
                self.num_layers += 1
            self.convs_output_shape = shape
        if len(linears) > 0:
            num_features = shape[0] if len(
                shape) == 1 else shape[0] * shape[1] * shape[2]
            self.flatten_output_shape = (num_features, )
            for h in linears:
                hidden_layer = nn.Linear(num_features, h)
                num_features = h
                self.linear_layers.append(hidden_layer)
                self.num_layers += 1
            self.linears_output_shape = (num_features, )
        if deconvs and len(deconvs) > 0:
            if len(linears) > 0:
                shape = unflatten_shape
            elif len(convs) > 0:
                pass  # shape = shape
            else:
                shape = unflatten_shape if len(shape) == 1 else shape
            self.unflatten_output_shape = shape
            for channels, kernel, stride in deconvs:
                deconv_layer = nn.ConvTranspose2d(
                    shape[0], channels, kernel, stride)
                shape = convT2d_shape_out(shape, channels, kernel, stride)
                self.deconv_layers.append(deconv_layer)
                self.num_layers += 1
            self.deconvs_output_shape = shape

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
        if not len(x.shape) == 2 and len(self.linear_layers) > 0:
            x = x.view(x.size(0), -1)
        for hidden_layer in self.linear_layers:
            x = hidden_layer(x)
            layers_remaining -= 1
            if layers_remaining > 0:
                x = torch.relu(x)
        if not len(x.shape) == 4 and len(self.deconv_layers) > 0:
            x = x.view(-1, *self.unflatten_output_shape)
        for deconv_layer in self.deconv_layers:
            x = deconv_layer(x)
            layers_remaining -= 1
            if layers_remaining > 0:
                x = torch.relu(x)
        return x
