import logging

import torch
from torch import nn

from RL.utils.util_fns import conv2d_shape_out, convT2d_shape_out, get_symmetric_deconvs


class FFModel(nn.Module):
    def __init__(self, input_shape, convs, linears, flatten=False, unflatten_shape=None, deconvs=[], act_fn_callable=lambda: nn.ReLU(), bn=False, ln=False, apply_act_output=False):
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
                if self.num_layers < total_layers or apply_act_output:
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

        num_features = None
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
                if self.num_layers < total_layers or apply_act_output:
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
                if self.num_layers < total_layers or apply_act_output:
                    if bn:
                        self.deconv_layers.append(nn.BatchNorm2d(shape[0]))
                    self.deconv_layers.append(act_fn_callable())
            self.deconvs_output_shape = shape
            self.output_shape = self.deconvs_output_shape

    def forward(self, x: torch.Tensor):
        '''should be a float tensor'''
        assert 2 <= len(x.shape) <= 4, f"shape of x is {x.shape}"
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


class Encoder(nn.Module):
    def __init__(self, input_shape, convs, embedding_size):
        super().__init__()
        self.encoder = FFModel(input_shape, convs, [
                               embedding_size], apply_act_output=True)

    def forward(self, x):
        return self.encoder(x)


class DecoderSymmetric(nn.Module):
    def __init__(self, input_shape, convs, embedding_size):
        super().__init__()
        if len(input_shape) > 1:
            deconvs, deconv_output_shape, convs_output_shape, flattened_shape, info = get_symmetric_deconvs(
                input_shape, convs)
            print(info)
            assert input_shape == deconv_output_shape, f"this conv configuration is not invertible, {input_shape}, {deconv_output_shape}"
            self.decoder = FFModel([embedding_size], [], [
                                   flattened_shape[0]], unflatten_shape=convs_output_shape, deconvs=deconvs)
        else:
            self.decoder = FFModel([embedding_size], [], [
                                   input_shape[0]])

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, convs, embedding_size):
        super().__init__()
        self.encoder = Encoder(input_shape, convs, embedding_size)
        self.decoder = DecoderSymmetric(input_shape, convs, embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class Discriminator(nn.Module):
    def __init__(self, input_shape, convs, embedding_size):
        super().__init__()
        # double the number of channels in conv1.. because input channels are doubled.
        if len(input_shape) > 1:
            convs = [(2 * convs[0][0], *convs[0][1:]), *convs[1:]]
            input_shape = (2 * input_shape[0], *input_shape[1:])
            # print(input_shape)
        self.discriminator = FFModel(input_shape, convs, [
                                     embedding_size, 2], act_fn_callable=lambda: nn.LeakyReLU())

    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)
        disc_logits = self.discriminator(x_cat)
        return disc_logits
