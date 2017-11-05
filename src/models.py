
import tensorflow as tf
from layers import *

def ConvNet(x, labels, c_num, batch_size, is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:

    pool_size=2
    pool=tf.nn.max_pool

    # conv1
    with tf.variable_scope('first', reuse=reuse):
      hidden_num = 32 
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num = hidden_num*2
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')


    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')


    with tf.variable_scope('last', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')
      x = tf.reshape(x, [batch_size, -1])

    feat = x

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc6', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables


def MobileNet(x, labels, c_num, batch_size, is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:

    pool_size=2
    pool=tf.nn.max_pool
    conv_factory_used=sepconv_factory

    # conv1
    with tf.variable_scope('first', reuse=reuse):
      hidden_num = 32
      x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      hidden_num = hidden_num*2
      x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')


    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')


    with tf.variable_scope('last', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
      x = pool(x, ksize=[1,pool_size,pool_size,1], strides=[1,2,2,1], padding='VALID')
      x = tf.reshape(x, [batch_size, -1])

    feat = x

    # dropout
#    if is_train:
#      x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc6', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables
