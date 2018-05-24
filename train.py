# coding:utf-8

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from torch.utils import data

from model import Generator, Discriminator
from dataset import get_dataset
from misc.utils import *


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
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
    parser.add_argument('--dataset', type = str, required = True,
                        help = 'Target dataset (like CelebA)')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    parser.add_argument('--n_epoch', type = int, default = 30,
                        help = '# of epochs [100]')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'Batch size [64]')
    parser.add_argument('--num_workers', type = int, default = 4,
                        help = '# of workers for dataloading [4]')

    parser.add_argument('--crop_type', type = str, default = 'center',
                        help = 'Crop type for raw images [center]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [128, 128],
                        help = 'Crop shape for raw data [128, 128]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = None,
                        help = 'Resize shape for raw data [None]')
    parser.add_argument('--resize_scale', type = float, default = None,
                        help = 'Resize scale for raw data [None]')
    parser.add_argument('--image_size', type = int, default = 128,
                        help = 'Image size to be processed 128')

    parser.add_argument('--z_dim', type = int, default = 128,
                        help = 'z (fake seed) dimension [128]')
    parser.add_argument('--n_critic', type = int, default = 5,
                        help = '# of critic training [5]')
    parser.add_argument('--lambda_', type = float, default = 10,
                        help = 'Grdient penalty coefficient [10]')

    parser.add_argument('--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint [None]')
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
    
