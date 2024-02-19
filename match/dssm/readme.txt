1.paper:2013,https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf
2.paper:<Learning Deep Structured Semantic Models for Web Search using Clickthrough Data>
3.其他：softmax的loss类似对比学习的infoNce loss
4.精排是特征的艺术，召回就是样本的艺术，dssm召回的负样本采样很重要，参考《Embedding-based Retrieval in Facebook Search》\ 《百度的莫比乌斯》
5.召回和粗排区别在于样本和目标，召回是全库负样本，粗排是曝光未点击负样本（和精排一致）
6.MIND关注用户召回阶段的多兴趣建模。他们提出采用胶囊网络的动态路由算法来获取用户的多兴趣表示。通过将用户的行为聚合成多个兴趣向量表示，每个兴趣向量代表不同的用户兴趣，再分别用这些向量去候选池中检索top k的物品。
