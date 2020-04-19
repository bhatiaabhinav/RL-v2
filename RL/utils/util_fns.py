import numpy as np


def conv2d_size_out(size, kernel, stride):
    return (size - (kernel - 1) - 1) // stride + 1


def convT2d_size_out(size, kernel, stride, padding=0, dilation=1):
    return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1


def conv2d_shape_out(shape, channels, kernel, stride):
    '''shape in c, h, w format'''
    if not hasattr(kernel, '__len__'):
        kernel = (kernel, kernel)
    if not hasattr(stride, '__len__'):
        stride = (stride, stride)
    return channels, conv2d_size_out(shape[1], kernel[0], stride[0]), conv2d_size_out(shape[2], kernel[1], stride[1])


def convT2d_shape_out(shape, channels, kernel, stride):
    '''shape in c, h, w format'''
    if not hasattr(kernel, '__len__'):
        kernel = (kernel, kernel)
    if not hasattr(stride, '__len__'):
        stride = (stride, stride)
    return channels, convT2d_size_out(shape[1], kernel[0], stride[0]), convT2d_size_out(shape[2], kernel[1], stride[1])


def toNpFloat32(x, expand_dims=False):
    '''Convert to numpy float32. Optionally add a batch axis'''
    if expand_dims:
        x = np.expand_dims(x, 0)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = x.astype(np.float32)
    return x
