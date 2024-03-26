1.paper https://www.ijcai.org/proceedings/2017/0239.pdf
2.<DeepFM: A Factorization-Machine based Neural Network for CTR Prediction>
3.DeepFM 是 Deep 与 FM 结合的产物，也是 Wide&Deep 的改进版，只是将其中的 LR 替换成了 FM，提升了模型 wide 侧提取信息的能力
4.embedding的两两点击和乘以离散0-1特征，本质是和所有离散特征点击，但是除了当前样本的特征类外其他都是0
