# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.compat.v1.get_variable_scope().name
    raise ValueError(
      "For the tensor `%s` in scope `%s`, the actual rank "
      "`%d` (shape = %s) is not equal to the expected rank `%s`" %
      (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


class EmbeddingSharedWeights(tf.keras.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size):
    """Specify characteristic parameters of embedding layer.

    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
    """
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

  def build(self, input_shape):
    """Build embedding layer."""
    with tf.name_scope("embedding_and_softmax"):
      # Create and initialize weights. The random normal initializer was chosen
      # arbitrarily, and works well.
      self.shared_weights = self.add_weight(
        "weights",
        shape=[self.vocab_size, self.hidden_size],
        dtype="float32",
        initializer=tf.random_normal_initializer(
          mean=0., stddev=self.hidden_size**-0.5))
    super(EmbeddingSharedWeights, self).build(input_shape)

  def get_config(self):
    return {
      "vocab_size": self.vocab_size,
      "hidden_size": self.hidden_size,
    }

  def call(self, inputs, mode="embedding"):
    """Get token embeddings of inputs.

    Args:
      inputs: An int64 tensor with shape [batch_size, length]
      mode: string, a valid value is one of "embedding" and "linear".
    Returns:
      outputs: (1) If mode == "embedding", output embedding tensor, float32 with
        shape [batch_size, length, embedding_size]; (2) mode == "linear", output
        linear tensor, float32 with shape [batch_size, length, vocab_size].
    Raises:
      ValueError: if mode is not valid.
    """
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == "linear":
      return self._linear(inputs)
    else:
      raise ValueError("mode {} is not valid.".format(mode))

  def _embedding(self, inputs):
    """Applies embedding based on inputs tensor."""
    with tf.name_scope("embedding"):
      # Create binary mask of size [batch_size, length]
      mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
      embeddings = tf.gather(self.shared_weights, inputs)
      embeddings *= tf.expand_dims(mask, -1)
      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size ** 0.5

      return embeddings

  def _linear(self, inputs):
    """Computes logits by running inputs through a linear layer.

    Args:
      inputs: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(inputs)[0]
      length = tf.shape(inputs)[1]

      x = tf.reshape(inputs, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])


class PositionalEmbeddingSharedWeights(tf.keras.layers.Layer):
  """Calculates positional embeddings and pre-softmax linear with shared weights."""

  def __init__(self, max_length, hidden_size, drop_out):
    """Specify characteristic parameters of embedding layer.

    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~73)
      max_length: int. Maximum sequence length that might ever be
        used with this model. This can be longer than the sequence length of
        input_tensor, but cannot be shorter.
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
    """
    super(PositionalEmbeddingSharedWeights, self).__init__()
    self.max_length = max_length
    self.hidden_size = hidden_size
    self.drop_out = drop_out


  def build(self, input_shape):
    """Build embedding layer."""
    # with tf.name_scope("pos_embedding_and_softmax"):
    #   # Create and initialize weights. The random normal initializer was chosen
    #   # arbitrarily, and works well.
    #   self.shared_emb_weights = self.add_weight(
    #     "emb_weights",
    #     shape=[self.vocab_size, self.hidden_size],
    #     dtype="float32",
    #     initializer=tf.random_normal_initializer(
    #       mean=0., stddev=self.hidden_size**-0.5))

    with tf.name_scope("pos_embedding_weights"):
      # Create and initialize weights. The random normal initializer was chosen
      # arbitrarily, and works well.
      self.shared_pos_weights = self.add_weight(
        "pos_weights",
        shape=[self.max_length, self.hidden_size],
        dtype="float32",
        initializer=tf.random_normal_initializer(
          mean=0., stddev=self.hidden_size**-0.5))
    super(PositionalEmbeddingSharedWeights, self).build(input_shape)

  def get_config(self):
    return {
      "vocab_size": self.vocab_size,
      "max_length": self.max_length,
      "hidden_size": self.hidden_size,
    }

  def call(self, inputs, mode="embedding"):
    """Get token embeddings of inputs.

    Args:
      inputs: An int64 tensor with shape [batch_size, length]
      mode: string, a valid value is one of "embedding" and "linear".
    Returns:
      outputs: (1) If mode == "embedding", output embedding tensor, float32 with
        shape [batch_size, length, embedding_size]; (2) mode == "linear", output
        linear tensor, float32 with shape [batch_size, length, vocab_size].
    Raises:
      ValueError: if mode is not valid.
    """

    """Applies embedding based on inputs tensor."""
    # with tf.name_scope("embedding"):
    #   # Create binary mask of size [batch_size, length]
    #   mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
    #   embeddings = tf.gather(self.shared_emb_weights, inputs)
    #   embeddings *= tf.expand_dims(mask, -1)
    #   # Scale embedding by the sqrt of the hidden size
    #   embeddings *= self.hidden_size ** 0.5
    #   embeddings = tf.cast(embeddings, self.params["dtype"])

    # return embeddings
    with tf.name_scope("pos_embedding"):
      seq_length = tf.shape(inputs)[1]
      position_embeddings = tf.slice(self.shared_pos_weights, [0, 0],
                                     [seq_length, -1])
      num_dims = len(inputs.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, self.hidden_size])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      encoder_inputs = inputs + position_embeddings

    return encoder_inputs



class EmbeddingFreezable(tf.keras.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, trainable, name=None):
    """Specify characteristic parameters of embedding layer.

    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
    """
    super(EmbeddingFreezable, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.trainable = trainable
    self.custom_name = name

  def build(self, input_shape):
    """Build embedding layer."""
    with tf.name_scope("embedding_and_softmax"):
      # Create and initialize weights. The random normal initializer was chosen
      # arbitrarily, and works well.
      self.shared_weights = self.add_weight(
        "weights",
        shape=[self.vocab_size, self.hidden_size],
        trainable=self.trainable,
        dtype="float32",
        initializer=tf.random_normal_initializer(
          mean=0., stddev=self.hidden_size**-0.5))
    super(EmbeddingFreezable, self).build(input_shape)

  def get_config(self):
    return {
      "vocab_size": self.vocab_size,
      "hidden_size": self.hidden_size,
    }

  def call(self, inputs, mode="embedding"):
    """Get token embeddings of inputs.

    Args:
      inputs: An int64 tensor with shape [batch_size, length]
      mode: string, a valid value is one of "embedding" and "linear".
    Returns:
      outputs: (1) If mode == "embedding", output embedding tensor, float32 with
        shape [batch_size, length, embedding_size]; (2) mode == "linear", output
        linear tensor, float32 with shape [batch_size, length, vocab_size].
    Raises:
      ValueError: if mode is not valid.
    """
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == "linear":
      return self._linear(inputs)
    else:
      raise ValueError("mode {} is not valid.".format(mode))

  def _embedding(self, inputs):
    """Applies embedding based on inputs tensor."""
    with tf.name_scope("embedding"):
      # Create binary mask of size [batch_size, length]
      mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
      embeddings = tf.gather(self.shared_weights, inputs)
      embeddings *= tf.expand_dims(mask, -1)
      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size ** 0.5

      return embeddings

  def _linear(self, inputs):
    """Computes logits by running inputs through a linear layer.

    Args:
      inputs: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(inputs)[0]
      length = tf.shape(inputs)[1]

      x = tf.reshape(inputs, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])