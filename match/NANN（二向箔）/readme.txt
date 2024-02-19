1.paper:2022,https://arxiv.org/pdf/2202.10226.pdf
2.paper:<Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation>
3.建库索引放到模型中去，避免预测和检索反复进行。
4.在通过近似最近邻搜索算法快速查找和用户相近的若干个商品时，使用深度神经网络模型的计算输出作为用户和商品的距离度量表示其相关性，替代内积、余弦相似度等度量形式的用户和商品的向量距离。这样，既可以充分使用模型的表达能力保证用户和商品相关性的准确性，也可以通过近似最近邻搜索算法（论文中使用HNSW算法）保证结果的快速返回。论文将该方案称为NANN（Neural Approximate Nearest Neighbor Search）。
