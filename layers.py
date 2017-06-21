# coding:utf-8

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

def leaky_relu(x, leak = 0.2):
    return tf.maximum(tf.minimum(0.0, leak * x), x)

def leaky_relu_batch_norm(x, leak = 0.2):
    return leaky_relu(batch_norm(x), leak)

def relu_batch_norm(x):
    return tf.nn.relu(batch_norm(x))
