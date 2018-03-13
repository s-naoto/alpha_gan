# coding: utf-8

import tensorflow as tf
from mnist_vae import MNISTEncoder
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.tensorboard.plugins import projector
from color_mnist import ColorMNIST

# model definition
encoder = MNISTEncoder(latent_size=50)
mnist = read_data_sets('./MNIST_data')
color_mnist = ColorMNIST(mnist.test, batch_size=64)

# calculation graph
x = color_mnist()
z = encoder.encode(x, is_train=False)


