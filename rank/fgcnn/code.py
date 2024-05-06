# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten

# 启用 Eager Execution 模式
tf.enable_eager_execution();


# 定义参数
batch_size = 2
seq_length = 3
embedding_size = 4
#
conv_kernel_widths=(7, 7, 7, 7);
conv_filters=(14, 16, 18, 20);
new_maps=(3, 3, 3, 3);
pooling_widths=(2, 2, 2, 2);
#
inputs = tf.random.uniform([batch_size, seq_length, embedding_size]);

pooling_result = tf.expand_dims(inputs, axis=3);
pooling_shape = pooling_result.get_shape().as_list();


def _conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Copy of the function of keras-team/keras because it's not in the public API
    So we can't use the function in keras-team/keras to test tf.keras

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

def _conv_output_shape(input_shape, kernel_size):
    # channels_last
    space = input_shape[1:-1]
    new_space = []
    for i in range(len(space)):
        new_dim = _conv_output_length(
            space[i],
            kernel_size[i],
            padding='same',
            stride=1,
            dilation=1)
        new_space.append(new_dim)
    return ([input_shape[0]] + new_space + [conv_filters])

def _pooling_output_shape(input_shape, pool_size):
    # channels_last

    rows = input_shape[1]
    cols = input_shape[2]
    rows = _conv_output_length(rows, pool_size[0], 'valid',
                                    pool_size[0])
    cols = _conv_output_length(cols, pool_size[1], 'valid',
                                    pool_size[1])
    return [input_shape[0], rows, cols, input_shape[3]]

new_feature_list = [];
for i in range(1, len(conv_filters) + 1):
    filters = conv_filters[i - 1]
    width = conv_kernel_widths[i - 1]
    new_filters = new_maps[i - 1]
    pooling_width = pooling_widths[i - 1]
    #
    conv_output_shape = _conv_output_shape(
            pooling_shape, (width, 1));
    pooling_shape = _pooling_output_shape(
            conv_output_shape, (pooling_width, 1));
    #
    pooling_result = Conv2D(
            filters=filters,
            kernel_size=(width, 1),
            strides=(1, 1),
            padding='same',
            activation='tanh', use_bias=True, )(pooling_result);
    pooling_result = MaxPooling2D(pool_size=(pooling_width, 1))(pooling_result);

    mid_layers = Flatten()(pooling_result);
    mid_layers = Dense(embedding_size,
                       activation='tanh', use_bias=True)(mid_layers);
    new_feature_list.append(mid_layers);
#
output = tf.concat(new_feature_list, 1);
#
print(new_feature_list)
print(output)
