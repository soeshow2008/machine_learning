import sys;
import numpy as np;
import tensorflow as tf;

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(64,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
]);
model.compile(optimizer='adam', loss='mean_squared_error');

for idx in range(1000):
    # random sampling
    X_new = np.random.rand(100, 64) * 10.0 - 5.0;
    y = model.predict(X_new)
    y = y.reshape(-1);
    # select opt
    fy = (y - 20000.0) * (y - 20000.0) + 5;
    min_idx = np.argsort(fy)[:10];
    min_y = y[min_idx];
    min_fy = fy[min_idx];
    print("xxxxx", idx, min_y[0], min_fy[0]);
    # train
    min_x = np.random.rand(10, 64) * 10.0 - 5.0;
    model.fit(min_x, min_y, epochs=10);

