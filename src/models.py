import tensorflow as tf
from layers import *

def quick_cnn(x, labels, c_num, batch_size, is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32 
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # Uncomment to see vinishing gradients
#    for l in range(8):
#      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
#        x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

    # fc4
    with tf.variable_scope('fc4', reuse=reuse):
      x = tf.reshape(x, [batch_size, -1])
      x = fc_factory(x, hidden_num, is_train, reuse)
    feat = x

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc5', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables
