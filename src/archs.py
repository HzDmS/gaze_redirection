# Network architectures.

from __future__ import division
from utils.ops import relu, conv2d, lrelu, instance_norm, deconv2d, tanh

import tensorflow as tf
import tensorflow.contrib.slim as slim


def discriminator(params, x_init, reuse=False):

    """ Discriminator.

    Parameters
    ----------
    params: dict.
    x_init: input tensor.
    reuse: bool, reuse the net if True.

    Returns
    -------
    x_gan: tensor, outputs for adversarial training.
    x_reg: tensor, outputs for gaze estimation.

    """

    layers = 5
    channel = 64
    image_size = params.image_size

    with tf.variable_scope('discriminator', reuse=reuse):

        # 64 3 -> 32 64 -> 16 128 -> 8 256 -> 4 512 -> 2 1024

        x = conv2d(x_init, channel, conv_filters_dim=4, d_h=2, d_w=2,
                   scope='conv_0', pad=1, use_bias=True)
        x = lrelu(x)

        for i in range(1, layers):
            x = conv2d(x, channel * 2, conv_filters_dim=4, d_h=2, d_w=2,
                       scope='conv_%d' % i, pad=1, use_bias=True)
            x = lrelu(x)
            channel = channel * 2

        filter_size = int(image_size / 2 ** layers)

        x_gan = conv2d(x, 1, conv_filters_dim=filter_size, d_h=1, d_w=1,
                       pad=1, scope='conv_logit_gan', use_bias=False)

        x_reg = conv2d(x, 2, conv_filters_dim=filter_size, d_h=1, d_w=1,
                       pad=0, scope='conv_logit_reg', use_bias=False)
        x_reg = tf.reshape(x_reg, [-1, 2])

        return x_gan, x_reg


def generator(input_, angles, reuse=False):

    """ Generator.

    Parameters
    ----------
    input_: tensor, input images.
    angles: tensor, target gaze direction.
    reuse: bool, reuse the net if True.

    Returns
    -------
    x: tensor, generated image.

    """

    channel = 64
    style_dim = angles.get_shape().as_list()[-1]

    angles_reshaped = tf.reshape(angles, [-1, 1, 1, style_dim])
    angles_tiled = tf.tile(angles_reshaped, [1, tf.shape(input_)[1],
                                             tf.shape(input_)[2], 1])
    x = tf.concat([input_, angles_tiled], axis=3)

    with tf.variable_scope('generator', reuse=reuse):

        # input layer
        x = conv2d(x, channel, d_h=1, d_w=1, scope='conv2d_input',
                   use_bias=False, pad=3, conv_filters_dim=7)
        x = instance_norm(x, scope='in_input')
        x = relu(x)

        # encoder
        for i in range(2):

            x = conv2d(x, 2 * channel, d_h=2, d_w=2, scope='conv2d_%d' % i,
                       use_bias=False, pad=1, conv_filters_dim=4)
            x = instance_norm(x, scope='in_conv_%d' % i)
            x = relu(x)
            channel = 2 * channel

        # bottleneck
        for i in range(6):

            x_a = conv2d(x, channel, conv_filters_dim=3, d_h=1, d_w=1,
                         pad=1, use_bias=False, scope='conv_res_a_%d' % i)
            x_a = instance_norm(x_a, 'in_res_a_%d' % i)
            x_a = relu(x_a)
            x_b = conv2d(x_a, channel, conv_filters_dim=3, d_h=1, d_w=1,
                         pad=1, use_bias=False, scope='conv_res_b_%d' % i)
            x_b = instance_norm(x_b, 'in_res_b_%d' % i)

            x = x + x_b

        # decoder
        for i in range(2):

            x = deconv2d(x, int(channel / 2), conv_filters_dim=4, d_h=2, d_w=2,
                         use_bias=False, scope='deconv_%d' % i)
            x = instance_norm(x, scope='in_decon_%d' % i)
            x = relu(x)
            channel = int(channel / 2)

        x = conv2d(x, 3, conv_filters_dim=7, d_h=1, d_w=1, pad=3,
                   use_bias=False, scope='output')
        x = tanh(x)

    return x


def vgg_16(inputs, scope='vgg_16', reuse=False):

    """ VGG-16.

    Parameters
    ----------
    inputs: tensor.
    scope: name of scope.
    reuse: reuse the net if True.

    Returns
    -------
    net: tensor, output tensor.
    end_points: dict, collection of layers.

    """

    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:

        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3],
                              scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)

    return net, end_points
