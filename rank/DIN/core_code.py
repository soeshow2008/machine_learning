def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False):
    '''
    query ：候选广告，shape: [B, H], 即i_emb；
    facts ：用户历史行为，shape: [B, T, H], 即h_emb，T是padding后的长度，每个长H的emb代表一个item；
    mask : Batch中每个行为的真实意义，shape: [B, H]；    
    '''     
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print ("querry_size mismatch")
        query = tf.concat(values = [
        query,
        query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])  
      
    # 转换mask  
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer，
    querry_size = query.get_shape().as_list()[-1] # H，这里是36
    
    # 1. 转换query维度，变成历史维度T 
    # query是[B, H]，转换到 queries 维度为(B, T, H)，为了让pos_item和用户行为序列中每个元素计算权重
    # 此时query是 Tensor("concat:0", shape=(?, 36), dtype=float32)
    # tf.shape(keys)[1] 结果就是 T，query是[B, H]，经过tile，就是把第一维按照 T 展开，得到[B, T * H] 
    queries = tf.tile(query, [1, tf.shape(facts)[1]]) # [B, T * H], 想象成贴瓷砖
    # 此时 queries 是 Tensor("Attention_layer/Tile:0", shape=(?, ?), dtype=float32)
    # queries 需要 reshape 成和 facts 相同的大小: [B, T, H]
    queries = tf.reshape(queries, tf.shape(facts)) # [B, T * H] -> [B, T, H]
    # 此时 queries 是 Tensor("Attention_layer/Reshape:0", shape=(?, ?, 36), dtype=float32) 
    
    # 2. 这部分目的就是为了在MLP之前多做一些捕获行为item和候选item之间关系的操作：加减乘除等。
    # 得到 Local Activation Unit 的输入。即 候选广告 queries 对应的 emb，用户历史行为序列 facts
    # 对应的 embed, 再加上它们之间的交叉特征, 进行 concat 后的结果
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1) # T*[B,H] ->[B, T, H]
    
    # 3. attention操作，通过几层MLP获取权重，这个DNN 网络的输出节点为 1
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
  	# 上一层 d_layer_3_all 的 shape 为 [B, T, 1]
 	  # 下一步 reshape 为 [B, 1, T], axis=2 这一维表示 T 个用户行为序列分别对应的权重参数    
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all # attention的输出, [B, 1, T]
    
    # 4. 得到有真实意义的score
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    # padding的mask后补一个很小的负数，这样后面计算 softmax 时, e^{x} 结果就约等于 0
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1) # 注意初始化为极小值
    # [B, 1, T] padding操作，为了忽略了padding对总体的影响，代码中利用tf.where将padding的向量(每个样本序列中空缺的商品)权重置为极小值(-2 ** 32 + 1)，而不是0
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # 5. Scale # attention的标准操作，做完scaled后再送入softmax得到最终的权重。
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # 6. Activation，得到归一化后的权重
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # 7. 得到了正确的权重 scores 以及用户历史行为序列 facts, 再进行矩阵相乘得到用户的兴趣表征
    # Weighted sum，
    if mode == 'SUM':
        # scores 的大小为 [B, 1, T], 表示每条历史行为的权重,
        # facts 为历史行为序列, 大小为 [B, T, H];
        # 两者用矩阵乘法做, 得到的结果 output 就是 [B, 1, H]
        # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
        # 这里的output是attention计算出来的权重，即论文公式(3)里的w，
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        # 从 [B, 1, H] 变化成 Batch * Time
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]]) 
        # 先把scores在最后增加一维，然后进行哈达码积，[B, T, H] x [B, T, 1] =  [B, T, H]
        output = facts * tf.expand_dims(scores, -1) 
        output = tf.reshape(output, tf.shape(facts)) # Batch * Time * Hidden Size
    return output
