# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d, conv2d_transpose, flatten

from layers import *

class Discriminator:

    def __init__(self, image_size = 64):
        self.image_size = image_size
        # self.x_dim = ????
        self.name = 'discriminator'

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            batch_size = tf.shape(x)[0]
            # x = tf.reshape(x, [batch_size, self.image_size, self.image_size, 3])

            conv1 = conv2d(x, 64, [4,4], [2,2],
                           weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                           activation_fn = leaky_relu)

            conv2 = conv2d(conv1, 128, [4,4], [2,2],
                           weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                           activation_fn = leaky_relu_batch_norm)

            conv3 = conv2d(conv2, 256, [4,4], [2,2],
                           weights_initializer= tf.random_normal_initializer(stddev = 0.02),
                           activation_fn = leaky_relu_batch_norm)

            conv4 = conv2d(conv3, 512, [4,4], [2,2],
                           weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                           activation_fn = leaky_relu_batch_norm)

            flat = flatten(conv4)
            fc = fully_connected(flat, 1, activation_fn = tf.identity)

            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator:
    
    def __init__(self, image_size = 64):
        self.image_size = image_size
        self.z_dim = 100
        # self.x_dim = ????
        self.name = 'generator'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            batch_size = tf.shape(z)[0]
            fc = fully_connected(z, 4*4*1024, activation_fn = tf.identity)

            conv1 = tf.reshape(fc, tf.stack([batch_size, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)

            conv2 = conv2d_transpose(conv1, 512, [4,4], [2,2],
                                     weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                                     activation_fn = relu_batch_norm)

            conv3 = conv2d_transpose(conv2, 256, [4,4], [2,2],
                                     weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                                     activation_fn = relu_batch_norm)

            conv4 = conv2d_transpose(conv3, 128, [4,4], [2,2],
                                     weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                                     activation_fn = relu_batch_norm)

            conv5 = conv2d_transpose(conv4, 3, [4,4], [2,2],
                                     weights_initializer = tf.random_normal_initializer(stddev = 0.02),
                                     activation_fn = tf.tanh)

            return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
            













    
