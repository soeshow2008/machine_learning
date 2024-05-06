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

inputs = tf.expand_dims(inputs, axis=3);
pooling_shape = inputs.get_shape().as_list();

new_feature_list = [];
for i in range(1, len(conv_filters) + 1):
    filters = conv_filters[i - 1]
    width = conv_kernel_widths[i - 1]
    new_filters = new_maps[i - 1]
    pooling_width = pooling_widths[i - 1]
    #
    mid_layers = Conv2D(
            filters=filters,
            kernel_size=(width, 1),
            strides=(1, 1),
            padding='same',
            activation='tanh', use_bias=True, )(inputs);
    mid_layers = MaxPooling2D(pool_size=(pooling_width, 1))(mid_layers);
    mid_layers = tf.reshape(mid_layers, [mid_layers.shape[0], -1]);
    mid_layers = Dense(embedding_size,
                       activation='tanh', use_bias=True)(mid_layers);
    new_feature_list.append(mid_layers);
#
output = tf.concat(new_feature_list, 1);
#
print(output)
