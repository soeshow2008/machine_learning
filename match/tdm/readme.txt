1.paper:,https://arxiv.org/pdf/1801.02294.pdf
2.paper:2018,<Learning Tree-based Deep Model for Recommender Systems>
3.支持任何复杂的模型（剥离dssm双塔结构限制，最后node或者item还是有向量的，只是可以做更复杂的上层网络）
4.查询算法：Beam Search
5.建库和模型训练两阶段方法：按照类目等初始化建库，模型训练正负样本（正样本为点击、负样本随机采样），item及他的所有父亲节点都需要训练，得到node和item的embedding；根据embedding重新聚类树(聚类方法可以是kmeans)；再训练模型再聚类树
6.JTM优化了kmeans建库和模型优化目标不一致问题，去掉了kmeans改用似然损失函数最小化方式建库。
