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
