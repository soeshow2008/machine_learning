fm_embeddings = tf.reshape(input_embedding, shape=[-1, field_num, embedding_size]);
squared_sum_embeddings = tf.reduce_sum(fm_embeddings, axis=1);
squared_sum_embeddings = tf.square(squared_sum_embeddings);
sum_squared_embeddings = tf.reduce_sum(tf.square(fm_embeddings), axis=1);
fm_value = tf.reduce_sum(0.5 * tf.subtract(squared_sum_embeddings, sum_squared_embeddings), axis=1);
fm_value = tf.reshape(fm_value, shape=[-1, 1]);
