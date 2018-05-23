# coding:utf-8

import os
import argparse
import numpy as np
import tensorflow as tf
from torch.utils import data

from model import Generator, Discriminator
from dataset import get_dataset
from misc.utils import *


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth() = True
        self.sess = tf.Session(config = config)
        self._build_dataloader()
        self._build_graph()

    def _build_graph(self):
        dataset = get_dataset(self.args.dataset)
        d_set = dataset(self.args.dataset_dir,
                        self.args.crop_type, self.args.crop_shape,
                        self.args.resize_shape, self.args.resize_scale)
        self.num_batches = int(len(d_set.samples)/self.args.batch_size)
        self.d_loader = data.DataLoader(d_set, shuffle = True,
                                        batch_size = self.args.batch_size,
                                        num_workers = self.args.num_workers,
                                        pin_memory = True)

    def _build_graph(self):
        self.z = tf.placeholder(tf.float32, shape = (None, self.args.z_dim),
                                name = 'z')
        self.x = tf.placeholder(tf.float32, shape = (None, self.args.image_size, self.args.image_size, 3),
                                name = 'x')

        self.gen = Generator(self.args.z_dim, self.args.image_size)
        self.disc = Discriminator(self.args.image_size)

        self.x_ = self.gen(self.z) # fake samples

        self.d_real = tf.reduce_mean(self.disc(self.x, reuse = False))
        self.d_fake = tf.reduce_mean(self.disc(self.x_))

        alpha = tf.random_uniform((self.args.batch_size, 1, 1, 1), minval = 0., maxval = 1.)
        x_interp = alpha*self.x + (1 - alpha)*self.x_
        gradients = tf.gradients(self.disc(x_interp), [x_interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 3))
        gradient_penalty = tf.reduce_mean((slopes -1.)**2)
        
        self.d_loss = self.d_fake - self.d_real + self.args.lambda_*gradient_penalty
        self.g_loss = -self.d_fake

        self.g_opt = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5, beta2 = 0.9)\
                             .minimize(self.g_loss, var_list = self.gen.vars)
        self.d_opt = tf.train.AdamOptimizer(learning_rate = 1e-4, beta1 = 0.5, beta2 = 0.9)\
                             .minimize(self.d_loss, var_list = self.disc.vars)

        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        for e in range(self.args.n_epoch):
            for i, images in enumerate(self.d_loader):
                
                images = images.numpy()/255.0
                
                for _ in range(self.args.n_critic):
                    randoms = np.random.uniform(-1, 1, (self.args.batch_size, self.args.z_dim))
                    _, d_real, d_fake = self.sess.run([self.d_opt, self.d_real, self.d_fake],
                                                      feed_dict = {self.z: randoms, self.x: images})

                randoms = np.random.uniform(-1, 1, (self.args.batch_size, self.args.z_dim))
                _ = self.sess.run(self.g_opt, feed_dict = {self.z: randoms})
                
                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, d_real-d_fake, None)

                if i%100 == 0:
                    randoms = np.random.uniform(-1, 1, (9, self.args.z_dim))
                    images_fake = self.sess.run(self.x_, feed_dict = {self.z: randoms})
                    images_fake = combine_images(images_fake)
                    images_fake = images_fake*127.5 + 127.5
                    if not os.path.exists('./sample'):
                        os.mkdir('./sample')
                    Image.fromarray(images_fake.astype(np.uint8))\
                        .save(f'./sample/sample_{str(e).zfill(3)}_{str(i).zfill(4)}.png')

            if not os.path.exists('./model'):
                od.mkdir('/model')
            self.saver.save(self.sess, f'./model/model_{str(e).zfill(3)}.ckpt')
                
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parset.add_argument('--dataset', type = str, required = True,
                        help = 'Target dataset (like CelebA)')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    ...
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
    
# def main():
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-e', '--epochs', type = int, default = 20,
#                         help = 'number of epochs [20]')
#     parser.add_argument('--train_size', type = int, default = np.inf,
#                         help = 'whole size of training data [all]')
#     parser.add_argument('--batch_size', type = int, default = 64,
#                         help = 'batch size [64]')
#     parser.add_argument('--nd', type = int, default = 5,
#                         help = 'discriminator training ratio [5]')
#     parser.add_argument('--target_size', type = int, default = 108,
#                         help = 'target area of original image')
#     parser.add_argument('--image_size', type = int, default = 64,
#                         help = 'size of generated images')
#     parser.add_argument('--datadir', type = str, nargs = '+', required = True,
#                         help = 'path to the directory contains training (image) data')
#     parser.add_argument('--split', type = int, default = 5,
#                         help = 'load data, with [5] split')
#     parser.add_argument('--sampledir', type = str, default = './image',
#                         help = 'path to the directory put generated samples [./image]')
#     parser.add_argument('--modeldir', type = str, default = './model',
#                         help = 'path to the directory parameter saved [./model]')
#     args = parser.parse_args()
    
#     sampler = InputSampler(datadir = args.datadir,
#                            target_size = args.target_size, image_size = args.image_size,
#                            split = args.split, num_utilize = args.train_size)
    
#     disc = Discriminator(image_size = args.image_size)
#     gen = Generator(image_size = args.image_size)
    
#     wgan = WassersteinGAN(gen, disc, args.nd, sampler)
#     wgan.train(batch_size = args.batch_size,
#                epochs = args.epochs,
#                sampledir = args.sampledir,
#                modeldir = args.modeldir)
    

# class WassersteinGAN:

#     def __init__(self, gen, disc, d_iter, sampler):

#         self.gen = gen
#         self.disc = disc
#         self.d_iter = d_iter
#         # self.x_dim = self.disc.x_dim
#         self.z_dim = self.gen.z_dim
#         self.sampler = sampler
#         self.image_size = self.gen.image_size

#         self.x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'x')
#         self.z = tf.placeholder(tf.float32, [None, self.z_dim], name = 'z')

#         self.x_ = self.gen(self.z)

#         self.d = self.disc(self.x, reuse = False)
#         self.d_ = self.disc(self.x_)

#         self.g_loss = tf.reduce_mean(self.d_)
#         self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

#         self.reg = apply_regularization(l1_regularizer(2.5e-5),
#                                         weights_list = [var for var in tf.global_variables() if 'weights' in var.name])

#         self.g_loss_reg = self.g_loss + self.reg
#         self.d_loss_reg = self.d_loss + self.reg

#         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#             self.d_opt = tf.train.RMSPropOptimizer(learning_rate = 5e-5)\
#                         .minimize(self.d_loss_reg, var_list = self.disc.vars)
#             self.g_opt = tf.train.RMSPropOptimizer(learning_rate = 5e-5)\
#                          .minimize(self.g_loss_reg, var_list = self.gen.vars)

#         self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc.vars]

#         self.saver = tf.train.Saver()
        
#         self.sess = tf.Session()

#     def train(self, batch_size = 64, epochs = 20,
#               sampledir = './image', modeldir = './model'):

#         num_batches = int(self.sampler.data_size/batch_size)
#         print('Number of batches : {}, epochs : {}'.format(num_batches, epochs))

#         # plt.ion()
#         self.sess.run(tf.global_variables_initializer())

#         for e in range(epochs):

#             for batch in range(num_batches):

#                 if batch in np.linspace(0, num_batches, self.sampler.split+1, dtype = int):
#                     self.sampler.reload()

#                 d_iter = self.d_iter
#                 if batch%500 == 0 or batch<25:
#                     d_iter = 100

#                 for _ in range(d_iter):
#                     bx = self.sampler.image_sample(batch_size)
#                     bz = self.sampler.noise_sample(batch_size, self.z_dim)
#                     self.sess.run(self.d_clip)
#                     self.sess.run(self.d_opt, feed_dict = {self.x: bx, self.z: bz})

#                 bz = self.sampler.noise_sample(batch_size, self.z_dim)
#                 self.sess.run(self.g_opt, feed_dict = {self.z: bz, self.x: bz})

                
#                 if batch%10 == 0:
#                     bx = self.sampler.image_sample(batch_size)
#                     bz = self.sampler.noise_sample(batch_size, self.z_dim)
#                     d_loss = self.sess.run(self.d_loss, feed_dict = {self.x: bx, self.z: bz})
#                     g_loss = self.sess.run(self.g_loss, feed_dict = {self.z: bz, self.x: bx})
                    
#                     print('epoch : {}, batch : {}, d_loss : {}, g_loss : {}'\
#                           .format(e, batch, d_loss - g_loss, g_loss))

#                 if batch%100 == 0:
#                     bz = self.sampler.noise_sample(batch_size, self.z_dim)
#                     fake_images = self.sess.run(self.x_, feed_dict = {self.z: bz})
#                     fake_images = combine_images(fake_images)
#                     fake_images = fake_images*127.5 + 127.5
#                     Image.fromarray(fake_images.astype(np.uint8))\
#                          .save(sampledir + 'sample_{}_{}.png'.format(e, batch))

                    
#             self.saver.save(self.sess, modeldir + 'model_{}epoch.ckpt'.format(e))
                    

# if __name__ == '__main__':
#     main()
