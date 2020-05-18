import logging

import torch
from torch import nn

from RL.utils.util_fns import conv2d_shape_out, convT2d_shape_out


class FFModel(nn.Module):
    def __init__(self, input_shape, convs, linears, flatten=False, unflatten_shape=None, deconvs=[], act_fn_callable=lambda: nn.ReLU(), bn=False, ln=False):
        super().__init__()
        self.input_shape = input_shape
        self.bn = bn

        self.num_layers = 0
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        self.flatten = flatten
        shape = (1, input_shape[0], input_shape[1]) if len(
            input_shape) == 2 else input_shape
        self.convs_output_shape = None
        self.flatten_output_shape = None
        self.linears_output_shape = None
        self.unflatten_output_shape = None
        self.deconvs_output_shape = None
        self.output_shape = input_shape
        self.act_fn_callable = act_fn_callable

        total_layers = len(convs) + len(linears) + len(deconvs)

        if len(input_shape) > 1:
            for channels, kernel, stride, padding in convs:
                conv_layer = nn.Conv2d(
                    shape[0], channels, kernel, stride, padding)
                self.conv_layers.append(conv_layer)
                shape = conv2d_shape_out(
                    shape, channels, kernel, stride, padding)
                self.num_layers += 1
                if self.num_layers < total_layers:
                    if bn:
                        self.conv_layers.append(nn.BatchNorm2d(shape[0]))
                    self.conv_layers.append(act_fn_callable())
            self.convs_output_shape = shape
            self.output_shape = self.convs_output_shape
        else:
            # Since skipping Conv layers, write a warning
            logging.getLogger(__name__).warn(
                f"Conv layers were requested but the input shape {input_shape} does not support convs")
            total_layers -= len(convs)

        if flatten or self.num_layers < total_layers:
            num_features = shape[0] if len(
                shape) == 1 else shape[0] * shape[1] * shape[2]
            self.flatten_output_shape = (num_features, )
            self.output_shape = self.flatten_output_shape

        if len(linears) > 0:
            for h in linears:
                hidden_layer = nn.Linear(num_features, h)
                self.linear_layers.append(hidden_layer)
                num_features = h
                self.num_layers += 1
                if self.num_layers < total_layers:
                    if ln:
                        self.linear_layers.append(nn.LayerNorm([num_features]))
                    elif bn:
                        self.linear_layers.append(nn.BatchNorm1d(num_features))
                    self.linear_layers.append(act_fn_callable())
            self.linears_output_shape = (num_features, )
            self.output_shape = self.linears_output_shape

        if len(deconvs) > 0:
            if len(linears) > 0:
                shape = unflatten_shape
            elif len(convs) > 0:
                pass  # shape = shape
            else:
                shape = unflatten_shape if len(shape) == 1 else shape
            self.unflatten_output_shape = shape
            for channels, kernel, stride, padding in deconvs:
                deconv_layer = nn.ConvTranspose2d(
                    shape[0], channels, kernel, stride, padding)
                self.deconv_layers.append(deconv_layer)
                shape = convT2d_shape_out(
                    shape, channels, kernel, stride, padding)
                self.num_layers += 1
                if self.num_layers < total_layers:
                    if bn:
                        self.deconv_layers.append(nn.BatchNorm2d(shape[0]))
                    self.deconv_layers.append(act_fn_callable())
            self.deconvs_output_shape = shape
            self.output_shape = self.deconvs_output_shape

    def forward(self, x: torch.Tensor):
        '''should be a float tensor'''
        assert 2 <= len(x.shape) <= 4
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        if not len(x.shape) == 2 and (self.flatten or len(self.linear_layers) > 0):
            x = x.view(x.size(0), -1)
        for hidden_layer in self.linear_layers:
            x = hidden_layer(x)
        if not len(x.shape) == 4 and len(self.deconv_layers) > 0:
            x = x.view(-1, *self.unflatten_output_shape)
        for deconv_layer in self.deconv_layers:
            x = deconv_layer(x)
        return x
