ipnn_embeddings = tf.reshape(input_embedding, shape=[-1, field_num, embedding_size]);
rows = [];
cols = [];
for i in range(field_num):
    for j in range(i + 1, field_num):
        rows.append(i);
        cols.append(j);
p = tf.gather(ipnn_embeddings, rows, axis=1);
q = tf.gather(ipnn_embeddings, cols, axis=1);
inner_product = tf.reduce_sum(p * q, axis=2);
ipnn_layer = tf.reshape(inner_product, shape=[-1, field_num * (deep_fid_num - 1) / 2]);
