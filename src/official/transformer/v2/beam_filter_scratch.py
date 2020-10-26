import tensorflow as tf



scores = tf.Variable([[0.8], [0.6], [0.2]]) # [None, 1]
py = tf.Variable([[2.,0.3, 0.4], [3.,0.2,0.3], [5.,0.8,0.1]])

prop_hat_list = []
prop_hat = tf.Variable([[1.,0.1, 0.2], [1.,0.3,0.1], [1.,0.5,0.6]]) # [None, 3]
prop_hat_list.append(prop_hat)
prop_hat = tf.Variable([[2.,0.2, 0.3], [2.,0.4,0.2], [2.,0.6,0.7]]) # [None, 3]
prop_hat_list.append(prop_hat)
prop_hat = tf.Variable([[3.,0.3, 0.4], [3.,0.5,0.3], [3.,0.7,0.8]]) # [None, 3]
prop_hat_list.append(prop_hat)
prop_hat = tf.Variable([[4.,0.4, 0.5], [4.,0.3,0.9], [9.,0.3,0.3]]) # [None, 3]
prop_hat_list.append(prop_hat)



beam_size = 4
ids = tf.Variable([[[1,2], [3,4], [5,6], [7,8]],
                  [[11,12], [13,14], [15,16], [17,18]],
                  [[21,22], [23,24], [25,26], [27,28]]]) # [None, beam_size, 2]
output_list = []
output_list_new = []
for i in range(beam_size):
    beam_score_list = []
    prop_hat = prop_hat_list[i]
    prop_diff = tf.math.abs(prop_hat - py)
    logp_diff = prop_diff[:, 0]  # [None, 0]
    qed_diff = prop_diff[:, 1]  # [None, 0]
    drd2_diff = prop_diff[:, 2]  # [None, 0]
    simnet = tf.expand_dims(qed_diff, axis=-1)
    simnet = tf.squeeze(simnet, axis=-1)
    beam_score_list.append((1-qed_diff) + (1-drd2_diff))
    beam_score_list.append(simnet)
    beam_score = tf.stack(beam_score_list, axis=-1)
    beam_score = tf.math.reduce_sum(beam_score, axis=-1)
    output_list_new.append(beam_score)
    output_list.append((1-qed_diff) + (1-drd2_diff)+simnet)

new_custom_score = tf.stack(output_list_new, axis=-1)
custom_score = tf.stack(output_list, axis=-1)

ranking = tf.argsort(custom_score, axis=-1, direction='DESCENDING')
# ranking = tf.argsort(custom_score, axis=-1, direction='ASCENDING')
# array([[2, 3, 1, 0],
#        [0, 1, 2, 3],
#        [3, 1, 0, 2]], dtype=int32)>

ids[0,2,:]
ids[1,2,:]
ids[2,3,:]

# <tf.Tensor: id=730, shape=(2,), dtype=int32, numpy=array([5, 6], dtype=int32)>
# >>> ids[1,0,:]
# <tf.Tensor: id=735, shape=(2,), dtype=int32, numpy=array([11, 12], dtype=int32)>
# >>> ids[2,3,:]
# <tf.Tensor: id=740, shape=(2,), dtype=int32, numpy=array([27, 28], dtype=int32)>


init_indices = tf.expand_dims(ranking[:,0],axis=-1)

indices = tf.stack(
    [tf.range(tf.shape(init_indices)[0]), tf.reshape(init_indices, [-1])],
    axis=1
)

top_decoded_ids = tf.gather_nd(ids, indices)









x = tf.keras.layers.Input((10,), dtype="int64", name="inputs")
tf.range(tf.shape(x)[0])

tf.stack(
    [tf.range(x.shape[0]), tf.reshape(init_indices, [-1])],
    axis=1
)

beam_size=4
scores = tf.keras.layers.Input((beam_size,), dtype="float32", name="inputs")

beam_score_list = []
beam_score_list.append(scores)
beam_score_list.append(scores)