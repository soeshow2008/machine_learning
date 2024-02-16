#paper:https://www.microsoft.com/en-us/download/details.aspx?id=52452
#paper:《Scalable Training of L1-Regularized Log-Linear Models》
#owlqn:微软提出，拟牛顿法，lbfgs+l1。batch训练(每次更新梯度是所有样本一起更新，sgd是一个/部分样本一次更新)。
#其他：
牛顿法：搜索方向+步长；
拟牛顿法：hessian逆近似运算，提升效率
sgd改进方法：addelta、adgrade、adam等一般是解决步长问题，自动学习率
