# coding:utf-8

import os
import time
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l1_regularizer

from model import Generator, Discriminator
from misc.dataIO import InputSampler
from misc.utils import *


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'number of epochs [20]')
    parser.add_argument('--train_size', type = int, default = np.inf,
                        help = 'whole size of training data [all]')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'batch size [64]')
    parser.add_argument('--nd', type = int, default = 5,
                        help = 'discriminator training ratio [5]')
    parser.add_argument('--target_size', type = int, default = 108,
                        help = 'target area of original image')
    parser.add_argument('--image_size', type = int, default = 64,
                        help = 'size of generated images')
    parser.add_argument('--datadir', type = str, nargs = '+', required = True,
                        help = 'path to the directory contains training (image) data')
    parser.add_argument('--split', type = int, default = 5,
                        help = 'load data, with [5] split')
    parser.add_argument('--sampledir', type = str, default = './image',
                        help = 'path to the directory put generated samples [./image]')
    args = parser.parse_args()
    
    sampler = InputSampler(datadir = args.datadir,
                           target_size = args.target_size, image_size = args.image_size,
                           split = args.split, num_utilize = args.train_size)
    
    disc = Discriminator(image_size = args.image_size)
    gen = Generator(image_size = args.image_size)
    
    wgan = WassersteinGAN(gen, disc, args.nd, sampler)
    wgan.train(batch_size = args.batch_size,
               epochs = args.epochs,
               sampledir = args.sampledir)
    

class WassersteinGAN:

    def __init__(self, gen, disc, d_iter, sampler):

        self.gen = gen
        self.disc = disc
        self.d_iter = d_iter
        # self.x_dim = self.disc.x_dim
        self.z_dim = self.gen.z_dim
        self.sampler = sampler
        self.image_size = self.gen.image_size

        self.x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'z')

        self.x_ = self.gen(self.z)

        self.d = self.disc(self.x, reuse = False)
        self.d_ = self.disc(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        self.reg = apply_regularization(l1_regularizer(2.5e-5),
                                        weights_list = [var for var in tf.global_variables() if 'weights' in var.name])

        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.RMSPropOptimizer(learning_rate = 5e-5)\
                        .minimize(self.d_loss_reg, var_list = self.disc.vars)
            self.g_opt = tf.train.RMSPropOptimizer(learning_rate = 5e-5)\
                         .minimize(self.g_loss_reg, var_list = self.gen.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc.vars]

        self.saver = tf.train.Saver()
        
        self.sess = tf.Session()

    def train(self, batch_size = 64, epochs = 20, sampledir = './sample'):

        num_batches = int(self.sampler.data_size/batch_size)
        print('Number of batches : {}, epochs : {}'.format(num_batches, epochs))

        # plt.ion()
        self.sess.run(tf.global_variables_initializer())

        for e in range(epochs):

            for batch in range(num_batches):

                if batch in np.linspace(0, num_batches, self.sampler.split+1, dtype = int):
                    self.sampler.reload()

                d_iter = self.d_iter
                if batch%500 == 0 or batch<25:
                    d_iter = 100

                for _ in range(d_iter):
                    bx = self.sampler.image_sample(batch_size)
                    bz = self.sampler.noise_sample(batch_size, self.z_dim)
                    self.sess.run(self.d_clip)
                    self.sess.run(self.d_opt, feed_dict = {self.x: bx, self.z: bz})

                bz = self.sampler.noise_sample(batch_size, self.z_dim)
                self.sess.run(self.g_opt, feed_dict = {self.z: bz, self.x: bz})

                
                if batch%10 == 0:
                    bx = self.sampler.image_sample(batch_size)
                    bz = self.sampler.noise_sample(batch_size, self.z_dim)
                    d_loss = self.sess.run(self.d_loss, feed_dict = {self.x: bx, self.z: bz})
                    g_loss = self.sess.run(self.g_loss, feed_dict = {self.z: bz, self.x: bx})
                    
                    print('epoch : {}, batch : {}, d_loss : {}, g_loss : {}'\
                          .format(e, batch, d_loss - g_loss, g_loss))

                if batch%100 == 0:
                    bz = self.sampler.noise_sample(batch_size, self.z_dim)
                    fake_images = self.sess.run(self.x_, feed_dict = {self.z: bz})
                    fake_images = combine_images(fake_images)
                    fake_images = fake_images*127.5 + 127.5
                    Image.fromarray(fake_images.astype(np.uint8))\
                         .save(sampledir + 'sample_{}_{}.png'.format(e, batch))

                    
            self.saver.save(self.sess, 'model_{}epoch.ckpt'.format(e))
                    

if __name__ == '__main__':
    main()
