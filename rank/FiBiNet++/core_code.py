import tensorflow as tf

inputs = tf.constant(
        [
            [[1,2,3], [4,5,6], [7,8,9], [11,12,13]],
            [[11,22,33], [44,55,66], [77,88,99],[111,112,113]]
            ]

        , tf.float32)
print("###inputs");
print(inputs);

# 假设输入特征的维度
batch_size = 2
field_size = 4
embedding_size = 3
reduction_ratio = 3  # 降维比率

# 输入占位符
input_features = inputs;#tf.placeholder(tf.float32, shape=[None, field_size, embedding_size], name='input_features')

# SENET层
# Squeeze: Global Average Pooling，对embedding_size维度求平均
squeeze = tf.reduce_mean(input_features, axis=-1, keepdims=True)
print("###squeeze", squeeze);
squeeze = tf.reshape(squeeze, shape=[-1, field_size]);
# Excitation: 两层全连接网络，第一层降维，第二层升维
reduced_size = max(1, field_size // reduction_ratio)
#excitation = tf.contrib.layers.fully_connected(inputs=squeeze,
#                                               num_outputs=reduced_size,
#                                               scope="senet_1",
#                                               activation_fn=tf.nn.relu,
#                                               );
excitation = tf.keras.layers.Dense(units=reduced_size,
                                   activation='relu',
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')(squeeze);
print("###excitation", excitation)
#excitation = tf.contrib.layers.fully_connected(inputs=excitation,
#                                               num_outputs=field_size * embedding_size,
#                                               scope="senet_2",
#                                               activation_fn=tf.nn.sigmoid,
#                                               );
excitation = tf.keras.layers.Dense(units=field_size * embedding_size,
                                   activation='sigmoid',
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')(excitation);
excitation = tf.reshape(excitation, shape=[-1, field_size, embedding_size]);
# 重新校准特征
scale = input_features * excitation
print("###input_features", input_features);
print("###excitation", excitation);
print("###scale", scale);
