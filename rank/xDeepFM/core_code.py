# 假设的输入维度
batch_size = 2
field_size = 4
embedding_size = 3
cin_layers = [field_size * embedding_size, field_size * embedding_size]  # CIN层的大小

# 输入占位符
x0 = inputs;
# 初始化CIN层的结果列表
cin_results = []

# 构建CIN层
hidden_layers_output = x0
for layer_size in cin_layers:
    # 计算外积矩阵
    # 使用广播机制计算所有外积
    outer_product_matrix = tf.matmul(
        tf.reshape(hidden_layers_output, [batch_size, field_size, 1, embedding_size]),
        tf.reshape(x0, [batch_size, 1, field_size, embedding_size]),
        transpose_b=True
    )
    #print(outer_product_matrix);
    outer_product_matrix = tf.reshape(outer_product_matrix, [batch_size, field_size * field_size])

    # 使用一个全连接层来压缩外积矩阵
    glorot = tf.initializers.glorot_normal()
    W = tf.Variable(glorot([field_size * field_size, layer_size]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([layer_size]), dtype=tf.float32)

    # 对embedding_size维度进行压缩
    print(tf.tensordot(outer_product_matrix, W, axes=1));
    hidden_layers_output = tf.nn.relu(tf.matmul(outer_product_matrix, W) + b);
    # 将当前层的输出添加到结果列表
    cin_results.append(hidden_layers_output)

# 将所有层的输出拼接起来
final_output = tf.concat(cin_results, axis=1)  # (batch_size, sum(cin_layers))




cin_embedding = tf.reshape(input_embedding, shape=[-1, field_num, embedding_size]);
cin_layers = [field_num * embedding_size, deep_fid_num * embedding_size];
x0 = cin_embedding;
cin_results = [];
hidden_layers_output = x0;
for layer_size in cin_layers:
    hidden_layers_output = tf.reshape(hidden_layers_output, shape=[-1, field_num, embedding_size]);
    outer_product_list = []
    for i in range(field_num):
        for j in range(field_num):
            vec_i = tf.reshape(hidden_layers_output[:, i, :], [batch_size, 1, embedding_size]);
            vec_j = tf.reshape(x0[:, j, :], [batch_size, embedding_size, 1]);
            outer_product = tf.matmul(vec_i, vec_j);
            outer_product_list.append(outer_product);
    outer_product_matrix = tf.concat(outer_product_list, axis=1);
    outer_product_matrix = tf.reshape(outer_product_matrix, [batch_size, field_num * field_num]);
    glorot = tf.initializers.glorot_normal();
    W = tf.Variable(glorot([field_num * deep_fid_num, layer_size]), dtype=tf.float32);
    b = tf.Variable(tf.zeros([layer_size]), dtype=tf.float32);
    hidden_layers_output = tf.nn.relu(tf.matmul(outer_product_matrix, W) + b);
    cin_results.append(hidden_layers_output);
xdeep_fm_layer = tf.concat(cin_results, axis=1);
