import numpy as np;
import matplotlib.pyplot as plt;
import random;
import math;

#np.random.seed(0);

FEATURE_DIM = 3;

#
b = 0.0;
a = 0.5;

def build_train_data(num):
    features = np.random.rand(num, FEATURE_DIM);
    for i in range(num):
        features[i][FEATURE_DIM - 1] = 1.0;
    labels = np.random.randint(0, 2, size=(num, 1));
    t_index = 0;
    for feature in features:
        if a * feature[0] + b > feature[1]:
            labels[t_index][0] = 1;
        else:
            labels[t_index][0] = 0;
        t_index += 1;

    train_data = np.hstack((features, labels));

    return train_data;

def sgn(x):
    if x > 0.0:
        return 1.0;
    elif x < 0.0:
        return -1.0;
    else:
        return 0.0;

def train_ftrl(train_data):
    epoches = 999;
    #
    alpha = 0.5;
    beta = 1.0;
    lambda1 = 1.0;
    lambda2 = 0.0;
    #
    zz = np.zeros(FEATURE_DIM);
    nn = np.zeros(FEATURE_DIM);
    ww = np.zeros(FEATURE_DIM);
    #
    for epoch in range(epoches):
        for element in train_data:
            sum_w = 0.0;
            for t_i in range(FEATURE_DIM):
                if abs(zz[t_i]) <= lambda1:
                    ww[t_i] = 0.0;
                else:
                    t_1 = -1.0 / ((beta + math.sqrt(nn[t_i])) / alpha + lambda2); 
                    t_2 = zz[t_i] - sgn(zz[t_i]) * lambda1;
                    ww[t_i] = t_1 * t_2;
                sum_w += element[t_i] * ww[t_i];
            pred = 1.0 / (1.0 + math.exp(-sum_w));
            for t_i in range(FEATURE_DIM):
                g = (pred - element[FEATURE_DIM]) * element[t_i];
                sigma = 1.0 / alpha * (math.sqrt(nn[t_i] + g * g) - math.sqrt(nn[t_i]));
                zz[t_i] = zz[t_i] + g - sigma * ww[t_i];
                nn[t_i] = nn[t_i] + g * g;
        print(ww);
    return ww;

def test_and_draw(train_data, model):
    print(model);
    # draw tarin data
    p_x = [];
    p_y = [];
    n_x = [];
    n_y = [];
    for element in train_data:
        if element[3] >= 1.0:
            p_x.append(element[0]);
            p_y.append(element[1]);
        else:
            n_x.append(element[0]);
            n_y.append(element[1]);
    plt.scatter(p_x, p_y, color='red');
    plt.scatter(n_x, n_y, color='blue');
    # draw split line
    plt.axline((0, b), slope=a, color='yellow')
    # test
    xx = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    yy = [];
    for t_x in xx:
        t_y = (-t_x * model[0] - model[2] - math.log(1.0)) / model[1];
        yy.append(t_y);
    plt.plot(xx, yy, color='black');
    # draw
    plt.show();

    return;

train_data = build_train_data(100);
model = train_ftrl(train_data);
test_and_draw(train_data, model);
