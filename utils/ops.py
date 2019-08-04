# This script contains all neural network layers and functions that are used
# the project.

from __future__ import division

import tensorflow as tf
import numpy as np

weight_init = tf.contrib.layers.xavier_initializer()


def instance_norm(x, scope='instance_norm'):

    """ Wrapper of instance normalization.

    Parameters
    ----------
    input: tensor.
    scope: name of the scope.

    Returns
    -------
    normalized tensor.

    """
    return tf.contrib.layers.instance_norm(
        x, epsilon=1e-05, center=True, scale=True, scope=scope)


def conv2d(input_, output_dim, d_h=2, d_w=2, scope='conv_0',
           conv_filters_dim=4, padding='zero', use_bias=True, pad=0):

    """ Wrapper of convolutional operation.

    Parameters
    ----------
    input_: a 4d tensor.
    output_dim: int, output channels.
    d_h: int, height of stride.
    d_w: int, width of stride.
    scope: str, name of variable scope.
    conv_filters_dim: int, size of kernel, width = height.
    padding: str, strategy of padding, one of "zero" and "reflect".
    use_bias: bool, whether to use bias in this layer.
    pad: int, size of padding.

    Returns
    -------
    conv: output 4d tensor.

    """

    k_initializer = tf.random_normal_initializer(stddev=0.02)
    b_initializer = tf.constant_initializer(0)
    k_h = k_w = conv_filters_dim

    with tf.variable_scope(scope):

        if padding == 'zero':
            x = tf.pad(
                input_,
                [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        elif padding == 'reflect':
            x = tf.pad(
                input_,
                [[0, 0], [pad, pad], [pad, pad], [0, 0]],
                mode='REFLECT')
        else:
            x = input_

        conv = tf.layers.conv2d(
            x,
            output_dim,
            kernel_size=[k_h, k_w],
            strides=(d_h, d_w),
            kernel_initializer=k_initializer,
            bias_initializer=b_initializer,
            use_bias=use_bias)

    return conv


def deconv2d(input_, output_dim, d_h=2, d_w=2, scope='deconv_0',
             conv_filters_dim=4, padding='SAME', use_bias=True):
    """Transposed convolution (fractional stride convolution) layer.

    Parameters
    ----------
    input_: tensor, input image.
    output_dim: int, number of channels.
    d_h: int, height of stride.
    d_w: int, width of stride.
    scope: str, name of scope.
    conv_filter_dim: int, kernel size.
    padding: int, "same" or "valid", case insensitive.
    use_bias: bool, use bias or not.

    Returns
    -------
    deconv: tensor, output tenosr.
    """

    k_initializer = tf.random_normal_initializer(stddev=0.02)
    b_initializer = tf.constant_initializer(0)
    k_h = k_w = conv_filters_dim

    deconv = tf.layers.conv2d_transpose(
        input_,
        output_dim,
        kernel_size=[k_h, k_w],
        strides=(d_h, d_w),
        padding=padding,
        kernel_initializer=k_initializer,
        bias_initializer=b_initializer,
        use_bias=use_bias,
        name=scope)

    return deconv


def relu(input_):
    """ Wrapper of ReLU function.

    Parameters
    ----------
    input_: tensor.

    Returns
    -------
    tensor.

    """
    return tf.nn.relu(input_)


def lrelu(input_):
    """ Wrapper of LeakyReLU function.

    Parameters
    ----------
    input_: tensor.

    Returns
    -------
    tensor.

    """
    return tf.nn.leaky_relu(input_, alpha=0.01)


def tanh(input_):
    """ Wrapper of tanh function.

    Parameters
    ----------
    input_: tensor.

    Returns
    -------
    tensor.

    """
    return tf.tanh(input_)


def l1_loss(x, y):

    """ L1 loss.

    Parameters
    ----------
    x: tensor.
    y: tensor, which should have the same shape as x.

    Returns
    -------
    loss: scalar, l1 loss.

    """

    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def l2_loss(x, y):

    """ L2 loss.

    Parameters
    ----------
    x: tensor
    y: tensor, which should have the same shape as x.

    Returns
    -------
    loss: scalar, l2 loss.

    """

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=[1, 2, 3]))

    return loss


def content_loss(endpoints_mixed, content_layers):

    """ Content loss.
    Ref: https://arxiv.org/abs/1603.08155.

    Parameters
    ----------
    endpoints_mixed: dict, (name, tensor).
    content_layers: list, name of layers used.

    Returns
    -------
    loss: scalar, content loss.

    """

    loss = 0
    for layer in content_layers:
        feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
        size = tf.size(feat_a)
        loss += tf.nn.l2_loss(feat_a - feat_b) * 2 / tf.to_float(size)

    return loss


def style_loss(endpoints_mixed, style_layers):

    """ Style loss.
    Ref: https://arxiv.org/abs/1603.08155.

    Parameters
    ----------
    endpoints_mixed: dict, (name, tensor).
    content_layers: list, name of layers used.

    Returns
    -------
    loss: scalar, style loss.

    """

    loss = 0
    for layer in style_layers:
        feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
        size = tf.size(feat_a)
        loss += tf.nn.l2_loss(
            gram(feat_a) - gram(feat_b)) * 2 / tf.to_float(size)

    return loss


def gram(layer):

    """ Compute gram matrix.
    Ref: https://arxiv.org/abs/1603.08155.

    Parameters
    ----------
    layer: tensor.

    Returns
    -------
    grams: gram matrices.

    """

    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    features = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    denominator = tf.to_float(width * height * num_filters)
    grams = tf.matmul(features, features, transpose_a=True) / denominator

    return grams


def angular2cart(angular):

    """ Angular coordinates to cartesian coordinates.

    Parameters
    ----------
    angular: list, [yaw, pitch]

    Returns
    -------
    np.array, coordinates in cartesian system.

    """

    theta = angular[:, 0] / 180.0 * np.pi
    phi = angular[:, 1] / 180.0 * np.pi
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    return np.stack([x, y, z], axis=1)


def angular_error(x, y):

    """Compute the angular error.

    Parameters
    ----------
    x: list, [yaw, pitch].
    y: list, [yaw, pitch].

    Returns
    -------
    int, error.

    """

    x = angular2cart(x)
    y = angular2cart(y)

    x_norm = np.sqrt(np.sum(np.square(x), axis=1))
    y_norm = np.sqrt(np.sum(np.square(y), axis=1))

    sim = np.divide(np.sum(np.multiply(x, y), axis=1),
                    np.multiply(x_norm, y_norm))

    sim = np.clip(sim, -1.0, 1.0)

    return np.arccos(sim) * 180.0 / np.pi
