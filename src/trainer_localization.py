from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange

from models import *

def norm_img(img):
  return img / 127.5 - 1.

def denorm_img(img):
  return (img + 1.) * 127.5

class Trainer(object):
  def __init__(self, config, data_loader, label_loader, test_data_loader, test_label_loader,
               train_location_loader, train_size_loader,  test_location_loader, test_size_loader,
               train_msk_loader, test_msk_loader, model):
    self.config = config
    self.data_loader = data_loader
    self.label_loader = label_loader
    self.test_data_loader = test_data_loader
    self.test_label_loader = test_label_loader

    self.optimizer = config.optimizer
    self.batch_size = config.batch_size
    self.batch_size_test = config.batch_size_test

    self.train_location_loader=train_location_loader
    self.train_size_loader=train_size_loader
    self.test_location_loader=test_location_loader
    self.test_size_loader=test_size_loader
    self.train_msk_loader=train_msk_loader
    self.test_msk_loader=test_msk_loader
    self.model=model

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.save_step = config.save_step
    self.test_iter = config.test_iter
    self.wd_ratio = config.wd_ratio

    self.lr = tf.Variable(config.lr, name='lr')

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*decay_factor, name='lr_update')

    self.c_num = config.c_num

    self.model_dir = config.model_dir
    self.load_path = config.load_path

    self.build_model()
    self.build_test_model()

    self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60,
                             global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    for step in trange(self.start_step, self.max_step):
      fetch_dict = {
        'c_optim': self.c_optim,
        'wd_optim': self.wd_optim,
        'c_loss': self.total_loss,
        'loss_cls': self.loss_cls,
        'loss_reg': self.loss_reg,
        'accuracy': self.accuracy,
        'x_cls':self.x_cls,
        'mask':self.train_msk_loader
      }

      if step % self.log_step == self.log_step - 1:
        fetch_dict.update({
          'lr': self.lr,
          'summary': self.summary_op })

      result = self.sess.run(fetch_dict)

      # print(result['mask'][0])
      # print(result['x_cls'][0])

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()

        lr = result['lr']
        c_loss = result['c_loss']
        loss_cls = result['loss_cls']
        loss_reg = result['loss_reg']
        accuracy = result['accuracy']

        print("\n[{}/{}:{:.6f}] Loss_cls: {:.6f} Loss_reg: {:.6f} Accuracy: {:.4f}" . \
              format(step, self.max_step, lr, loss_cls, loss_reg, accuracy))
        sys.stdout.flush()

      if step % self.save_step == self.save_step - 1:
        self.saver.save(self.sess, self.model_dir + '/model')

        test_accuracy = 0
        test_regression = 0
        for iter in np.arange(self.test_iter):
          fetch_dict = { "test_accuracy":self.test_accuracy, 'loss_reg': self.test_loss_reg, }
          result = self.sess.run(fetch_dict)
          test_accuracy += result['test_accuracy']
          test_regression += result['loss_reg']
        test_accuracy /= self.test_iter
        test_regression /= self.test_iter

        print("\n[{}/{}:{:.6f}] Test Accuracy: {:.4f} Loss_reg: {:.4f}" . \
              format(step, self.max_step, lr, test_accuracy, test_regression))
        sys.stdout.flush()


      if step % self.epoch_step == self.epoch_step - 1:
        self.sess.run([self.lr_update])

  def build_model(self):
    self.x = self.data_loader
    self.labels = self.label_loader
    x = (self.x)

    self.total_loss, self.loss_cls, self.loss_reg, feat, self.accuracy, self.c_var, self.x_cls = self.model(
      x,self.train_location_loader,self.train_size_loader,self.train_msk_loader, self.labels, self.c_num, self.batch_size, is_train=True, reuse=False)
    self.total_loss = tf.reduce_mean(self.total_loss)
    self.loss_cls = tf.reduce_mean(self.loss_cls)
    self.loss_reg = tf.reduce_mean(self.loss_reg)

    # Gather gradients of conv1 & fc4 weights for logging
    with tf.variable_scope("C/first", reuse=True):
      conv1_weights = tf.get_variable("weights")
    conv1_grad = tf.reduce_max(tf.abs(tf.gradients(self.total_loss, conv1_weights, self.total_loss)))

    with tf.variable_scope("C/last", reuse=True):
      fc4_weights = tf.get_variable("weights")
    fc4_grad = tf.reduce_max(tf.abs(tf.gradients(self.total_loss, fc4_weights, self.total_loss)))

    x_grad = tf.gradients(self.total_loss, x, self.total_loss)
    x_grad = tf.reduce_sum(tf.abs(x_grad[0]), 3, True)
    x_grad = (x_grad - tf.reduce_min(x_grad)) / (tf.reduce_max(x_grad) - tf.reduce_mean(x_grad))
    x_grad = tf.multiply(self.x , x_grad)

    wd_optimizer = tf.train.GradientDescentOptimizer(self.lr)
    if self.optimizer == 'sgd':
      c_optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
    elif self.optimizer == 'adam':
      c_optimizer = tf.train.AdamOptimizer(self.lr)
    else:
      raise Exception("[!] Caution! Don't use {} opimizer.".format(self.optimizer))

    for var in tf.trainable_variables():
      weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
      tf.add_to_collection('losses', weight_decay)
    wd_loss = tf.add_n(tf.get_collection('losses'))

    self.c_optim = c_optimizer.minimize(self.total_loss, var_list=self.c_var)
    self.wd_optim = wd_optimizer.minimize(wd_loss)

    self.summary_op = tf.summary.merge([
      tf.summary.scalar("total_loss", self.total_loss),
      tf.summary.scalar("loss_cls", self.loss_cls),
      tf.summary.scalar("loss_reg", self.loss_reg),
      tf.summary.scalar("accuracy", self.accuracy),
      tf.summary.scalar("lr", self.lr),
      tf.summary.scalar("conv1_grad", conv1_grad),
      tf.summary.scalar("fc4_grad", fc4_grad),

      tf.summary.image("inputs", self.x),
      tf.summary.image("x_grad", x_grad),

      tf.summary.histogram("feature", feat)
    ])

  def test(self):
    self.saver.restore(self.sess, self.model_dir)
    test_accuracy = 0
    for iter in trange(self.test_iter):
      fetch_dict = {"test_accuracy":self.test_accuracy}
      result = self.sess.run(fetch_dict)
      test_accuracy += result['test_accuracy']
    test_accuracy /= self.test_iter

    print("Accuracy: {:.4f}" . format(test_accuracy))

  def build_test_model(self):
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader
    test_x = (self.test_x)

    self.test_total_loss, self.test_loss_cls, self.test_loss_reg, self.test_feat, self.test_accuracy, var, self.test_x_cls = self.model(
      test_x,self.test_location_loader,self.test_size_loader,self.test_msk_loader, self.test_labels, self.c_num, self.batch_size_test, is_train=False, reuse=True)
    self.test_total_loss = tf.reduce_mean(self.test_total_loss)
    self.test_loss_cls = tf.reduce_mean(self.test_loss_cls)
    self.test_loss_reg = tf.reduce_mean(self.test_loss_reg)

