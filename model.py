# coding:utf-8
import tensorflow as tf
import numpy as np


class Discriminator(object):
    def __init__(self, image_size, name = 'discriminator'):
        self.image_size = image_size
        self.name = 'discriminator'

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            x = tf.layers.Conv2D(64, (5, 5), (2, 2), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(128, (5, 5), (2, 2), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(256, (5, 5), (2, 2), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(512, (5, 5), (2, 2), 'same')(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.flatten(x)
            
            return tf.layers.Dense(1)(x)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
    

class Generator(object):
    def __init__(self, z_dim, image_size, name = 'generator'):
        self.z_dim = z_dim
        assert np.log2(image_size) == int(np.log2(image_size))
        self.image_size = image_size
        self.name = name

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            x = tf.layers.Dense(4*4*1024)(z)
            x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
            x = tf.reshape(x, (-1, 4, 4, 1024))
            
            num_channel = 1024
            for _ in range(int(np.log2(self.image_size/4))-1):
                num_channel /= 2
                x = tf.layers.Conv2DTranspose(int(num_channel), (5, 5), (2, 2), 'same')(x)
                x = tf.layers.BatchNormalization()(x)
                x = tf.nn.relu(x)
                
            x = tf.layers.Conv2DTranspose(3, (5, 5), (2, 2), 'same')(x)
            return tf.nn.tanh(x)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
            













    
