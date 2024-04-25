def split_heads(x, batch_size, num_heads, depth):
    """分拆最后一个维度到 (num_heads, depth).
    转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3]);

def muti_head_attention(v, k, q, q_mask, k_mask, num_units, num_heads, sname):
    depth = num_units // num_heads;
    batch_size = tf.shape(q)[0]

    q = tf.layers.dense(q, num_units, activation=None, name="intQ_{}".format(sname));
    k = tf.layers.dense(k, num_units, activation=None, name="intK_{}".format(sname));
    v = tf.layers.dense(v, num_units, activation=None, name="intV_{}".format(sname));

    q = split_heads(q, batch_size, num_heads, depth)  # (batch_size, num_heads, seq_len_q, depth)
    k = split_heads(k, batch_size, num_heads, depth)  # (batch_size, num_heads, seq_len_k, depth)
    v = split_heads(v, batch_size, num_heads, depth)  # (batch_size, num_heads, seq_len_v, depth)

    ori_q_mask = q_mask;
    q_mask = split_heads(q_mask, batch_size, num_heads, depth)  # (batch_size, num_heads, seq_len_v, depth)
    k_mask = split_heads(k_mask, batch_size, num_heads, depth)  # (batch_size, num_heads, seq_len_v, depth)
    qk_mask = tf.matmul(q_mask, k_mask, transpose_b=True);
    qk_mask = tf.where(tf.equal(qk_mask, tf.zeros_like(qk_mask)),
            tf.ones_like(qk_mask, dtype=tf.float32), tf.zeros_like(qk_mask, dtype=tf.float32))

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, qk_mask);

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, num_units))  # (batch_size, seq_len_q, d_model)

    output = tf.layers.dense(concat_attention, num_units, activation=None, name="intOut_{}".format(sname));
    output = output * ori_q_mask;

    return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # 将 sin 应用于数组中的偶数索引（indices）；2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # 将 cos 应用于数组中的奇数索引；2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def att_pooling_common(seq_emb,ad_emb,agg_emb,batch_size,ad_emb_size,hidden_size,is_training,mask,sname="1"):
    #(target_all, user_all);
    user_all = ad_emb[1];
    ad_emb = ad_emb[0];
    #
    inputs = tf.concat([tf.reshape(ad_emb, [-1, 1, ad_emb_size]), seq_emb], 1);
    element_mask = tf.where(tf.not_equal(inputs, tf.zeros_like(inputs)),
            tf.ones_like(inputs, dtype=tf.float32), tf.zeros_like(inputs, dtype=tf.float32))
    inputs_shape = inputs.get_shape().as_list();
    seq_len = inputs_shape[1];
    #
    pos_encoding = positional_encoding(100, ad_emb_size);
    inputs *= tf.math.sqrt(tf.cast(ad_emb_size, tf.float32))
    inputs += pos_encoding[:, :seq_len, :]
    #
    outputs = inputs;
    for i in range(6):
        values = outputs;
        keys = outputs;
        querys = outputs;
        (outputs, _) = muti_head_attention(values, keys, querys, element_mask, element_mask, ad_emb_size, 2, "mulatt" + sname + "_" + str(i));
        outputs += querys;
        outputs_1 = tf.contrib.layers.layer_norm(outputs, scope="intnorm_" + sname + "_" + str(i));
        #ffn
        outputs = tf.layers.dense(outputs_1, ad_emb_size, activation=tf.nn.relu, name="ffn_1_{}_{}".format(sname, i));
        outputs = tf.layers.dense(outputs, ad_emb_size, activation=None, name="ffn_2_{}_{}".format(sname, i));
        outputs += outputs_1;
        outputs = tf.contrib.layers.layer_norm(outputs, scope="intnorm_1_" + sname + "_" + str(i));

    first_embedding = tf.strided_slice(outputs, [0, 0, 0], [batch_size, 1, ad_emb_size], [1, 1, 1]);
    first_embedding = tf.reshape(first_embedding, [-1, ad_emb_size]);

    return first_embedding;

