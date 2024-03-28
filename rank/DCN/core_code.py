x0 = input_embedding;
x_cross = x0;
for i in range(3):
    x_w = tf.contrib.layers.fully_connected(inputs=x_cross,
            num_outputs=1,
            );
    x_cross = x0 * x_w + x_cross
dcn_layer = x_cross;
