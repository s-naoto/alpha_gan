# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import sonnet as snt

from alpha_gan import BaseModel


# Encoder
class MNISTEncoder(BaseModel):
    def __init__(self, name='MNIST_Encoder', latent_size=10, regularization=1.e-4):
        super(MNISTEncoder, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization),
               'b': l2_regularizer(scale=regularization)}

        with self._enter_variable_scope():
            self.conv1 = snt.Conv2D(name='conv2d_1', output_channels=32, kernel_shape=5, stride=2, regularizers=reg)
            self.bn1 = snt.BatchNorm(name='batch_norm_1')
            self.conv2 = snt.Conv2D(name='conv2d_2', output_channels=64, kernel_shape=5, stride=1, regularizers=reg)
            self.bn2 = snt.BatchNorm(name='batch_norm_2')
            self.conv3 = snt.Conv2D(name='conv2d_3', output_channels=64, kernel_shape=5, stride=2, regularizers=reg)
            self.bn3 = snt.BatchNorm(name='batch_norm_3')
            self.flatten = snt.BatchFlatten(name='flatten')
            self.mean = snt.Linear(name='mean', output_size=latent_size, regularizers=reg)
            self.variance = snt.Linear(name='variance', output_size=latent_size, regularizers=reg)

    def _build(self, x, is_train=False):
        h = tf.nn.relu(self.conv1(x))
        h = self.bn1(h, is_train)
        h = tf.nn.relu(self.conv2(h))
        h = self.bn2(h, is_train)
        h = tf.nn.relu(self.conv3(h))
        h = self.bn3(h, is_train)
        h = self.flatten(h)
        mu = self.mean(h)
        s = self.variance(h)
        return mu, s

    def encode(self, x, is_train=False):
        with tf.name_scope('encode'):
            mu, s = self(x, is_train)
            z = mu + s * tf.random_normal(shape=mu.shape)
        return z


# Generator
class MNISTGenerator(BaseModel):
    def __init__(self, name='MNIST_Generator', regularization=1.e-4):
        super(MNISTGenerator, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization),
               'b': l2_regularizer(scale=regularization)}

        with self._enter_variable_scope():
            self.linear = snt.Linear(name='linear', output_size=3136, regularizers=reg)
            self.bn1 = snt.BatchNorm(name='batch_norm_1')
            self.reshape = snt.BatchReshape(name='reshape', shape=[7, 7, 64])
            self.deconv1 = snt.Conv2DTranspose(name='tr-conv2d_1', output_channels=64,
                                               kernel_shape=5, stride=2, regularizers=reg)
            self.bn2 = snt.BatchNorm(name='batch_norm_2')
            self.deconv2 = snt.Conv2DTranspose(name='tr-conv2d_2', output_channels=32,
                                               kernel_shape=5, stride=1, regularizers=reg)
            self.bn3 = snt.BatchNorm(name='batch_norm_3')
            self.deconv3 = snt.Conv2DTranspose(name='tr-conv2d_3', output_channels=3,
                                               kernel_shape=5, stride=2, regularizers=reg)

    def _build(self, z, is_train=False):
        h = tf.nn.relu(self.linear(z))
        h = self.bn1(h, is_train)
        h = self.reshape(h)
        h = tf.nn.relu(self.deconv1(h))
        h = self.bn2(h, is_train)
        h = tf.nn.relu(self.deconv2(h))
        h = self.bn3(h, is_train)
        h = self.deconv3(h)
        return h


# Discriminator
class MNISTDiscriminator(BaseModel):
    def __init__(self, name='MNIST_Discriminator', regularization=1.e-4):
        super(MNISTDiscriminator, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization),
               'b': l2_regularizer(scale=regularization)}

        with self._enter_variable_scope():
            self.conv1 = snt.Conv2D(name='conv2d_1', output_channels=8, kernel_shape=5, stride=2, regularizers=reg)
            self.bn1 = snt.BatchNorm(name='batch_norm_1')
            self.conv2 = snt.Conv2D(name='conv2d_2', output_channels=16, kernel_shape=5, stride=1, regularizers=reg)
            self.bn2 = snt.BatchNorm(name='batch_norm_2')
            self.conv3 = snt.Conv2D(name='conv2d_3', output_channels=32, kernel_shape=5, stride=2, regularizers=reg)
            self.bn3 = snt.BatchNorm(name='batch_norm_3')
            self.conv4 = snt.Conv2D(name='conv2d_4', output_channels=64, kernel_shape=5, stride=1, regularizers=reg)
            self.bn4 = snt.BatchNorm(name='batch_norm_4')
            self.conv5 = snt.Conv2D(name='conv2d_5', output_channels=65, kernel_shape=5, stride=2, regularizers=reg)
            self.bn5 = snt.BatchNorm(name='batch_norm_5')
            self.flatten = snt.BatchFlatten(name='flatten')
            self.linear = snt.Linear(name='l', output_size=1, regularizers=reg)

    def _build(self, x, is_train=False):
        h = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
        h = self.bn1(h, is_train)
        h = tf.nn.leaky_relu(self.conv2(h), alpha=0.2)
        h = self.bn2(h, is_train)
        h = tf.nn.leaky_relu(self.conv3(h), alpha=0.2)
        h = self.bn3(h, is_train)
        h = tf.nn.leaky_relu(self.conv4(h), alpha=0.2)
        h = self.bn4(h, is_train)
        h = tf.nn.leaky_relu(self.conv5(h), alpha=0.2)
        h = self.bn5(h, is_train)
        h = self.flatten(h)
        h = self.linear(h)
        return tf.nn.sigmoid(h)
