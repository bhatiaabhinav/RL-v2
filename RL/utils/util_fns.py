import math
import json
import numpy as np


def conv2d_size_out(size, kernel, stride, padding):
    return (size + 2 * padding - (kernel - 1) - 1) // stride + 1


def convT2d_size_out(size, kernel, stride, padding=0, dilation=1):
    return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1


def conv2d_shape_out(shape, channels, kernel, stride, padding):
    '''shape in c, h, w format'''
    if not hasattr(kernel, '__len__'):
        kernel = (kernel, kernel)
    if not hasattr(stride, '__len__'):
        stride = (stride, stride)
    if not hasattr(padding, '__len__'):
        padding = (padding, padding)
    return channels, conv2d_size_out(shape[1], kernel[0], stride[0], padding[0]), conv2d_size_out(shape[2], kernel[1], stride[1], padding[1])


def convT2d_shape_out(shape, channels, kernel, stride, padding):
    '''shape in c, h, w format'''
    if not hasattr(kernel, '__len__'):
        kernel = (kernel, kernel)
    if not hasattr(stride, '__len__'):
        stride = (stride, stride)
    if not hasattr(padding, '__len__'):
        padding = (padding, padding)
    return channels, convT2d_size_out(shape[1], kernel[0], stride[0], padding[0]), convT2d_size_out(shape[2], kernel[1], stride[1], padding[1])


def get_symmetric_deconvs(input_shape, convs):
    prints = [['input', input_shape]]
    shape = input_shape
    for layer in convs:
        shape = conv2d_shape_out(shape, *layer)
        prints.append(['conv', layer, shape])
    conv_last_shape = shape
    flattened_shape = (shape[0] * shape[1] * shape[2],)
    conv_channels, conv_kernals, conv_strides, conv_pads = zip(*convs)
    convt_channels = list(conv_channels[::-1][1:]) + [input_shape[0]]
    convt_kernals = conv_kernals[::-1]
    convt_strides = conv_strides[::-1]
    convt_pads = conv_pads[::-1]
    deconvs = list(zip(convt_channels, convt_kernals,
                       convt_strides, convt_pads))
    for layer in deconvs:
        shape = convT2d_shape_out(shape, *layer)
        prints.append(['conv2d', layer, shape])
    prints.append(['output', shape])
    return deconvs, shape, conv_last_shape, flattened_shape, prints


def toNpFloat32(x, expand_dims=False):
    '''Convert to numpy float32. Optionally add a batch axis'''
    if expand_dims:
        x = np.expand_dims(x, 0)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = x.astype(np.float32)
    return x


def update_mean(mean, n, x, remove=False):
    """Calculates the updated mean of a collection on adding or removing an element. Returns 0 mean if updated number of elements become 0.

    Args:
        mean (float): current mean of the collection
        n (int): current number of elements in the collection
        x (float): the element to be either added or removed
        remove (bool, optional): Whether to add or remove the element. Defaults to False.

    Returns:
        float, int: updated mean, number of elements
    """
    mask = 2 * int(not remove) - 1
    new_n = n + mask
    if new_n == 0:
        return 0.0, 0
    if new_n < 0:
        raise ValueError('Cannot remove an element from an empty collection')
    sum_x = mean * n
    new_sum_x = sum_x + mask * x
    new_mean = new_sum_x / new_n
    return new_mean, new_n


def update_mean_std(mean, std, n, x, remove=False):
    """Calculates the updated mean and std of a collection on adding or removing an element. Returns 0 mean and std if updated number of elements become 0.

    Args:
        mean (float): current mean
        std (float): current std
        n (int): current number of elements in the collection
        x (float): the element to be either added or removed
        remove (bool, optional): Whether to add or remove the element. Defaults to False.

    Returns:
        float, float, int: updated mean, std, number of elements

    This method uses the property that
        variance
        = E[(X-u)^2]
        = E[X^2] - (E[X])^2
    """

    new_mean, new_n = update_mean(mean, n, x, remove=remove)
    if new_n == 0:
        return 0, 0, 0
    variance = std ** 2
    mean_x_squared = variance + mean ** 2  # by E[X^2] = variance + (E[X])^2
    new_mean_x_squared, new_n = update_mean(
        mean_x_squared, n, x**2, remove=remove)
    new_variance = new_mean_x_squared - new_mean**2
    if new_variance < -0.1:
        print("precision error? variance", new_variance)
    new_variance = max(0, new_variance)  # a defence againt neg variance
    new_std = math.sqrt(new_variance)
    return new_mean, new_std, new_n


def update_mean_std_corr(mean_x, mean_y, std_x, std_y, corr, n, x, y, remove=False):
    """Calculates the updated means, standard dev, and Pearson's correlation coefficient of collections X and Y on adding or removing a pair of elements x and y. Returns 0 means, stds and corr if updated number of elements become 0.

    Args:
        mean_x (float): current E[X}
        mean_y (float): current E[Y]
        std_x (float): current Std(X)
        std_y (float): current Std(Y)
        corr (float): current Corr(X, Y)
        n (int): = number of elements in collection X and Y
        x (float): element x to be either added to or removed from collection X
        y (float): element y to be either added to or removed from collection Y
        remove (bool, optional): Whether to add or remove x and y. Defaults to False.

    Returns:
        float, float, float, float, float, int: updated E[X], E[Y], Std(X), Std(Y), Corr(X, Y), n

    This method uses the property that:
    cov(X, Y)
    = corr(X, Y) * Std(X) * Std(Y)
    = E[XY] - E[X]E[Y]
    """
    new_mean_x, new_std_x, new_n = update_mean_std(
        mean_x, std_x, n, x, remove=remove)
    if new_n == 0:
        return 0, 0, 0, 0, 0, 0
    new_mean_y, new_std_y, new_n = update_mean_std(
        mean_y, std_y, n, y, remove=remove)
    cov = corr * std_x * std_y
    mean_xy = cov + mean_x * mean_y
    new_mean_xy, new_n = update_mean(mean_xy, n, x * y, remove=remove)
    new_cov = new_mean_xy - new_mean_x * new_mean_y
    if abs(new_cov) < 1e-9:
        new_corr = 0.0
    else:
        new_corr = new_cov / (new_std_x * new_std_y)
    if abs(new_corr) > 1 + 1e-9:
        print('Precision error? Corr', new_corr)
    new_corr = min(max(-1, new_corr), 1)
    return new_mean_x, new_mean_y, new_std_x, new_std_y, new_corr, new_n


def save(filename, data):
    with open(filename, "w") as file:
        json.dump(data, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)
