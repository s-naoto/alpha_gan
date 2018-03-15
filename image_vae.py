# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import sonnet as snt

from alpha_gan import BaseModel


class Encoder(BaseModel):
    def __init__(self, name='Encoder', latent_size=50, image_size=64, ndf=64, regularization=1.e-4):
        super(Encoder, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization)}

        self.convs = []
        self.batch_norms = []
        self.latent_size = latent_size

        csize, cndf = image_size / 2, ndf

        with self._enter_variable_scope():
            self.convs.append(snt.Conv2D(name='conv2d_1',
                                         output_channels=ndf,
                                         kernel_shape=4,
                                         stride=2,
                                         padding='SAME',
                                         regularizers=reg,
                                         use_bias=False))
            self.batch_norms.append(snt.BatchNorm(name='batch_norm_1'))

            n_layer = 2
            while csize > 4:
                self.convs.append(snt.Conv2D(name='conv2d_{}'.format(n_layer),
                                             output_channels=cndf * 2,
                                             kernel_shape=4,
                                             stride=2,
                                             padding='SAME',
                                             regularizers=reg,
                                             use_bias=False))
                self.batch_norms.append(snt.BatchNorm(name='batch_norm_{}'.format(n_layer)))
                cndf = cndf * 2
                csize = csize // 2

            self.mean = snt.Conv2D(name='conv_mean',
                                   output_channels=latent_size,
                                   kernel_shape=4,
                                   stride=1,
                                   padding='VALID',
                                   regularizers=reg,
                                   use_bias=False)
            self.variance = snt.Conv2D(name='conv_variance',
                                       output_channels=latent_size,
                                       kernel_shape=4,
                                       stride=1,
                                       padding='VALID',
                                       regularizers=reg,
                                       use_bias=False)

    def _build(self, x, is_train=False):
        h = x
        for c, bn in zip(self.convs, self.batch_norms):
            h = tf.nn.relu(bn(c(h), is_train))
        m = tf.reduce_mean(self.mean(h), axis=[1, 2])
        v = tf.reduce_mean(self.variance(h), axis=[1, 2])
        return m, v

    def encode(self, x):
        with tf.name_scope('encode'):
            m, v = self(x, False)
            z = m + v * tf.random_normal(shape=m.shape)
        return z


class Generator(BaseModel):
    def __init__(self, name='Generator', latent_size=50, image_size=64, ngf=64, regularization=1.e-4):
        super(Generator, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization)}

        self.conv_trs = []
        self.batch_norms = []
        self.latent_size = latent_size

        cngf, tisize = ngf // 2, 4
        while tisize != image_size:
            cngf = cngf * 2
            tisize = tisize * 2

        with self._enter_variable_scope():
            self.reshape = snt.BatchReshape(name='batch_reshape', shape=[1, 1, latent_size])
            self.conv_trs.append(snt.Conv2DTranspose(name='tr-conv2d_1',
                                                     output_channels=cngf,
                                                     kernel_shape=4,
                                                     stride=1,
                                                     padding='VALID',
                                                     regularizers=reg,
                                                     use_bias=False))
            self.batch_norms.append(snt.BatchNorm(name='batch_norm_1'))
            csize, cndf = 4, cngf
            n_layer = 2
            while csize < image_size // 2:
                self.conv_trs.append(snt.Conv2DTranspose(name='tr-conv2d_{}'.format(n_layer),
                                                         output_channels=cndf // 2,
                                                         kernel_shape=4,
                                                         stride=2,
                                                         padding='SAME',
                                                         regularizers=reg,
                                                         use_bias=False))
                self.batch_norms.append(snt.BatchNorm(name='batch_norm_{}'.format(n_layer)))
                n_layer += 1
                cndf = cndf // 2
                csize = csize * 2

            self.conv_trs.append(snt.Conv2DTranspose(name='tr-conv2d_{}'.format(n_layer),
                                                     output_channels=3,
                                                     kernel_shape=4,
                                                     stride=2,
                                                     padding='SAME',
                                                     regularizers=reg,
                                                     use_bias=False))

    def _build(self, z, is_train=False):
        h = self.reshape(z)
        for dc, b in zip(self.conv_trs, self.batch_norms):
            h = tf.nn.relu(b(dc(h), is_train))
        h = self.conv_trs[-1](h)
        return tf.nn.tanh(h)


class Discriminator(BaseModel):
    def __init__(self, name='Discriminator', image_size=64, ndf=64, regularization=1.e-4):
        super(Discriminator, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization)}

        self.convs = []
        self.batch_norms = []

        csize, cndf = image_size / 2, ndf

        with self._enter_variable_scope():
            self.convs.append(snt.Conv2D(name='conv2d_1',
                                         output_channels=ndf,
                                         kernel_shape=4,
                                         stride=2,
                                         padding='VALID',
                                         regularizers=reg,
                                         use_bias=False))
            self.batch_norms.append(snt.BatchNorm(name='batch_norm_1'))

            n_layer = 2
            while csize > 4:
                self.convs.append(snt.Conv2D(name='conv2d_{}'.format(n_layer),
                                             output_channels=cndf * 2,
                                             kernel_shape=4,
                                             stride=2,
                                             padding='SAME',
                                             regularizers=reg,
                                             use_bias=False))
                self.batch_norms.append(snt.BatchNorm(name='batch_norm_{}'.format(n_layer)))
                cndf = cndf * 2
                csize = csize // 2
                n_layer += 1

            self.convs.append(snt.Conv2D(name='conv2d_{}'.format(n_layer),
                                         output_channels=1,
                                         kernel_shape=4,
                                         stride=1,
                                         padding='SAME',
                                         regularizers=reg,
                                         use_bias=False))

    def _build(self, x, is_train=False):
        h = x
        for c, bn in zip(self.convs, self.batch_norms):
            h = tf.nn.leaky_relu(bn(c(h), is_train), alpha=0.2)
        h = tf.reduce_mean(self.convs[-1](h), axis=[1, 2])
        return tf.nn.sigmoid(h)


if __name__ == '__main__':
    import numpy as np

    x_real = tf.constant(np.random.rand(10, 128, 128, 3), dtype=tf.float32)

    e = Encoder(image_size=128)
    g = Generator(image_size=128)
    d = Discriminator(image_size=128)

    mu, s = e(x_real, is_train=False)
    z_hat = mu + s * tf.random_normal(shape=mu.shape)
    x_rec = g(z_hat, is_train=False)
    z_gen = tf.random_normal(shape=mu.shape)
    x_gen = g(z_gen, is_train=False)
    f_real = d(x_real, is_train=False)
    f_rec = d(x_rec, is_train=False)
    f_gen = d(x_gen, is_train=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        image, latent, reconstruct, generated = sess.run([x_real, z_hat, x_rec, x_gen])
        print(latent.shape, reconstruct.shape, generated.shape)
