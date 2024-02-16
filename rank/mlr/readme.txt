#paper:https://arxiv.org/pdf/1704.05194.pdf
#paper:《Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction》
#MLR:MLR(Mixed Logistic Regression, 混合逻辑斯特回归)
原理：分治原理(注意是样本划分而不是特征类划分，这里举个例子是不同的用户分层可以使用不同的lr区域)，x学习多个lr、x学习多个softmax分类权重，然后线性加权最终值

