1.paper https://www.ijcai.org/proceedings/2017/0239.pdf
2.<DeepFM: A Factorization-Machine based Neural Network for CTR Prediction>
3.DeepFM 是 Deep 与 FM 结合的产物，也是 Wide&Deep 的改进版，只是将其中的 LR 替换成了 FM，提升了模型 wide 侧提取信息的能力
4.FM最终输出是一个值，不是向量，表达能力弱一些
5.embedding的两两点击和乘以离散0-1特征，本质是和所有离散特征点击，但是除了当前样本的特征类外其他都是0
6.批量多特征类的向量内积，公式推导，可以转换成更容易矩阵计算的方式，和的平方-平方的和
7.人工交叉特征相比自动交叉特征，人工表达能力强泛化能力弱，自动可能相反？
