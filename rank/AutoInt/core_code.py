import tensorflow as tf

# 假设的输入维度
batch_size = 32
field_size = 10
embedding_size = 8
num_heads = 2  # 多头注意力的头数
num_layers = 3  # 注意力层的数量
dropout_rate = 0.1  # Dropout比率

# 输入占位符
inputs = tf.placeholder(tf.float32, shape=[None, field_size, embedding_size], name='input_features')

# 自注意力机制
queries = inputs
keys = inputs
num_units = queries.get_shape().as_list()[-1]  # 注意力机制的输出维度

# 多头注意力
for i in range(num_layers):
    with tf.variable_scope("layer_{}".format(i)):
        # 线性变换
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        # 分割成多头
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # 计算注意力权重
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        outputs = tf.nn.softmax(outputs)

        # 应用注意力权重
        outputs = tf.matmul(outputs, V_)

        # 恢复原始的头数
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        
        # 残差连接
        outputs += queries
        
        # 层归一化
        outputs = tf.contrib.layers.layer_norm(outputs)

        # 准备下一层的输入
        queries = outputs

# 输出层
outputs = tf.reshape(outputs, [-1, field_size * embedding_size])
logits = tf.layers.dense(outputs, 1)

# 创建会话并初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 示例输入数据
example_input_features = tf.random.normal([batch_size, field_size, embedding_size])
logits_eval = sess.run(logits, feed_dict={inputs: example_input_features.eval(session=sess)})
print("AutoInt logits shape:", logits_eval.shape)

