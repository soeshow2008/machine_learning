1. 更新公式 Q（s，a） = r  + γ * max（Q（s+1， a））；不同于Q-learning的Q(s, a) = Q(s, a) + α * （ r  + γ * max（Q（s+1， a）） - Q(s, a)）
2. 是off-policy，因为有Eplison-greey的探索，使得学习策略和探索策略不是同一个（看着别人学，而不是自己探索自己学）。
3. 经验回放：训练稳定性，避免依赖关系是的DNN训练走偏（随机梯度依赖随机样本）
4. Target-Q（只用于计算Q值用于Q—NET训练）：lazy模型，每隔一段时间复制Q-net的参数，仅用于计算Q（s+1，a）用于Q-net的label。也是为了训练稳定性，使得训练的参数和label尽量隔开。
5. DoubleDQN：max（Q（s+1， a））计算label时，最优action是用Q-net计算得到而不是Target-Q获得。
