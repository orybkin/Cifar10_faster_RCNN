import tensorflow as tf
from layers import *
from spatial_transformer import transformer


def ConvNet(x, labels, c_num, batch_size, is_train, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool

        # conv1
        with tf.variable_scope('first', reuse=reuse):
            hidden_num = 32
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        with tf.variable_scope('conv2', reuse=reuse):
            hidden_num = hidden_num * 2
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        with tf.variable_scope('conv3', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv4', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('last', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
            x = tf.reshape(x, [batch_size, -1])

        feat = x

        # local5
        with tf.variable_scope('fc6', reuse=reuse):
            W = tf.get_variable('weights', [hidden_num, c_num],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.matmul(x, W)

        # Softmax
        with tf.variable_scope('sm', reuse=reuse):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

    variables = tf.contrib.framework.get_variables(vs)
    return loss, feat, accuracy, variables


def MobileNet(x, labels, c_num, batch_size, is_train, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool
        conv_factory_used = sepconv_factory

        # conv1
        with tf.variable_scope('first', reuse=reuse):
            hidden_num = 32
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        with tf.variable_scope('conv2', reuse=reuse):
            hidden_num = hidden_num * 2
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        with tf.variable_scope('conv3', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv4', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('last', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
            x = tf.reshape(x, [batch_size, -1])

        feat = x

        # dropout
        #    if is_train:
        #      x = tf.nn.dropout(x, keep_prob=0.5)

        # local5
        with tf.variable_scope('fc6', reuse=reuse):
            W = tf.get_variable('weights', [hidden_num, c_num],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.matmul(x, W)

        # Softmax
        with tf.variable_scope('sm', reuse=reuse):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

    variables = tf.contrib.framework.get_variables(vs)
    return loss, feat, accuracy, variables


def ResNet(x, labels, c_num, batch_size, is_train, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool
        conv_factory_used = resblock_factory

        # conv1
        with tf.variable_scope('first', reuse=reuse):
            hidden_num = 32
            x = conv_factory_used(x, hidden_num, 3, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        for l in range(3):
            hidden_num = hidden_num * 2
            with tf.variable_scope('rd_conv_' + str(l), reuse=reuse):
                x = conv_factory_used(x, hidden_num, 3, 1, is_train, reuse)
                x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('last', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')
            x = tf.reshape(x, [batch_size, -1])

        feat = x

        # dropout
        #    if is_train:
        #      x = tf.nn.dropout(x, keep_prob=0.5)

        # local5
        with tf.variable_scope('fc6', reuse=reuse):
            W = tf.get_variable('weights', [hidden_num, c_num],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.matmul(x, W)

        # Softmax
        with tf.variable_scope('sm', reuse=reuse):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

    variables = tf.contrib.framework.get_variables(vs)
    return loss, feat, accuracy, variables


def RCNN(x, location, size, mask, labels, c_num, batch_size, is_train, reuse, conv_factory_used=conv_factory):
    with tf.variable_scope('C', reuse=reuse) as vs:
        pool_size = 2
        pool = tf.nn.max_pool

        image = x

        with tf.variable_scope('first', reuse=reuse):
            hidden_num = 32
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv2', reuse=reuse):
            hidden_num = hidden_num * 2
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv3', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory_used(x, hidden_num, 5, 1, is_train, reuse)
            x = pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('last', reuse=reuse):
            hidden_num = 2 * hidden_num
            x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

        x_feat = x

        with tf.variable_scope('RPN', reuse=reuse):
            with tf.variable_scope('conv1', reuse=reuse):
                inter_x = conv_factory(x_feat, hidden_num, 3, 1, is_train, reuse)

            with tf.variable_scope('cls', reuse=reuse):
                with tf.variable_scope('conv', reuse=reuse):
                    W = tf.get_variable('weights', [1, 1, hidden_num, 1],
                                        initializer=tf.contrib.layers.variance_scaling_initializer())
                    b = tf.get_variable('biases', [1, 1, 1, 1],
                                        initializer=tf.constant_initializer(0.0))
                    x_cls = tf.nn.conv2d(inter_x, W, strides=[1, 1, 1, 1], padding='SAME') + b

                with tf.variable_scope('loss', reuse=reuse):
                    print(x_cls)
                    print(tf.equal(mask, 0))
                    loss_mask = tf.not_equal(mask, 2)[:, :, :,
                                0]  # white areas of the image are those which should not contribute to the loss

                    loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_cls,
                                                                       labels=tf.cast(tf.equal(mask, 1), tf.float32))
                    loss_cls = tf.boolean_mask(loss_cls, loss_mask)
                    # loss_cls = tf.where(white_mask, tf.constant(0, tf.float32, [batch_size,6,6]), loss_cls)
                    x_cls_1 = tf.sigmoid(x_cls)
                    chosen = tf.argmax(tf.reshape(x_cls_1, [batch_size, -1, 1]), axis=1)
                    print('chosen', chosen)
                    x_cls = tf.cast(tf.round(x_cls_1), tf.int64)
                    accuracy = (tf.to_float(tf.equal(x_cls, mask)))
                    accuracy = tf.boolean_mask(accuracy, loss_mask)
                    # accuracy = tf.where(white_mask, tf.constant(0, tf.float32, [batch_size,6,6]), accuracy)
                    accuracy = tf.reduce_mean(accuracy)

            with tf.variable_scope('reg', reuse=reuse):
                with tf.variable_scope('conv', reuse=reuse):
                    W = tf.get_variable('weights', [1, 1, hidden_num, 3],
                                        initializer=tf.contrib.layers.variance_scaling_initializer())
                    b = tf.get_variable('biases', [1, 1, 1, 3],
                                        initializer=tf.constant_initializer([24, 24, 32]))
                    x_reg = tf.nn.conv2d(inter_x, W, strides=[1, 1, 1, 1], padding='SAME') + b

                with tf.variable_scope('loss', reuse=reuse):
                    print(x_reg)
                    print(tf.equal(mask, 0))
                    print(size)
                    print(location)

                    loss_mask = tf.equal(mask, 1)[:, :, :, 0]

                    def enlarge(tensor):
                        return tf.tile(tf.reshape(tensor, [batch_size, 1, 1]), [1, 6, 6])

                    # x_reg_t=tf.transpose(tf.reshape(tf.transpose(x_reg, [0, 3, 1, 2]), [batch_size, 3, -1]), [0, 2, 1])
                    # print('reshaped', tf.reshape(x_reg, [batch_size, -1, 3]))
                    x_reg_t=tf.reshape(x_reg, [batch_size, -1, 3])
                    chosen_reg = tf.gather_nd(x_reg_t, tf.concat([tf.expand_dims(tf.range(batch_size, dtype=tf.int64), 1), chosen[:, 0:1]], axis=1))
                    print('chose', chosen_reg)

                    t_x = (x_reg[:, :, :, 0] - enlarge(location[:, 0])) / 32
                    t_y = (x_reg[:, :, :, 1] - enlarge(location[:, 1])) / 32
                    t_w = tf.log(x_reg[:, :, :, 2] / enlarge(size))
                    loss_reg = smooth_l1(t_w) + smooth_l1(t_x) + smooth_l1(t_y)

                    loss_reg = tf.boolean_mask(loss_reg,
                                               loss_mask)  # tf.where(loss_mask, loss_reg, tf.constant(0, tf.float32, [batch_size,6,6]))
                    loss_reg = tf.reduce_mean(loss_reg)

        theta = tf.transpose(tf.convert_to_tensor(
            [chosen_reg[:, 2] / 48, tf.constant(0, tf.float32, shape=[batch_size]), (chosen_reg[:, 1] - 24) / 24,
             tf.constant(0, tf.float32, shape=[batch_size]), chosen_reg[:, 2] / 48, (chosen_reg[:, 0] - 24) / 24]))
        cropped_images = transformer(image, theta, [32, 32])
        x = transformer(x_feat, theta, [4, 4])

        with tf.variable_scope('conv5', reuse=reuse):
            x = conv_factory(x, hidden_num, 3, 1, is_train, reuse, hidden_num, [batch_size, 4, 4, hidden_num])
            x = tf.reshape(x, [batch_size, -1])

        with tf.variable_scope('fc6', reuse=reuse):
            W = tf.get_variable('weights', [hidden_num * 4 * 4, c_num],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.matmul(x, W)

        # Softmax
        with tf.variable_scope('sm', reuse=reuse):
            loss_cls_final = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
            accuracy_final = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))
            loss_cls_final = tf.reduce_mean(loss_cls_final)

    loss = loss_reg + loss_cls # + loss_cls_final
    variables = tf.contrib.framework.get_variables(vs)
    return loss, loss_cls, loss_reg, loss_cls_final, x_feat, accuracy, accuracy_final, variables, [x_cls_1, chosen,chosen_reg,location,size], cropped_images
