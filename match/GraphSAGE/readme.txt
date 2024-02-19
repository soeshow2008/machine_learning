1.paper:2017,https://arxiv.org/abs/1706.02216
2.paper:<Inductive Representation Learning on Large Graphs>
3.从两个方面对传统的GCN做了改进，一是在训练时的，采样方式将GCN的全图采样优化到部分以节点为中心的邻居抽样，这使得大规模图数据的分布式训练成为可能，并且使得网络可以学习没有见过的节点，这也使得GraphSAGE可以做归纳学习（Inductive Learning）。第二点是GraphSAGE研究了若干种邻居聚合的方式，并通过实验和理论分析对比了不同聚合方式的优缺点。
