# coding: utf-8

import tensorflow as tf
from mnist_vae import MNISTEncoder
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.tensorboard.plugins import projector
from color_mnist import ColorMNIST
from PIL import Image
import numpy as np
import math


def make_sprite(images, output_path):
    # get image size
    w, h = images[0].size
    m_h = m_w = math.floor(np.sqrt(len(images))) + 1

    # make master image
    # number of images in a row and a column must be "same"
    master = Image.new(
        mode='RGB',
        size=(m_w * w, m_h * h),
        color=(0, 0, 0)
    )

    # paste images to master image
    for i, img in enumerate(images):
        px = (i % m_w) * w
        py = (i // m_w) * h
        master.paste(img, (px, py))

    # output master image
    master.save(output_path)


# model definition
encoder = MNISTEncoder(latent_size=50)
mnist = read_data_sets('./MNIST_data')
color_mnist = ColorMNIST(mnist.test, batch_size=64)

# calculation graph
x = color_mnist()
z = encoder.encode(x, is_train=False)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(encoder.variable_list)
    saver.restore(sess, 'ColorMNIST/alpha_gan_model.ckpt')

    imgs = []
    feats = []
    for _ in range(10):
        x_real, feat = sess.run([x, z])
        imgs.append(x_real)
        feats.append(feat)

    images = np.r_[imgs]
    features = np.r_[feats]

    summary_writer = tf.summary.FileWriter('encoder')
    tensor = tf.Variable(features, trainable=False, name='extracted_features')
    sess.run(tf.variables_initializer([tensor]))
    ft_saver = tf.train.Saver([tensor])
    ft_saver.save(sess, "encoder/emb.ckpt")

    # projectorの設定
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor.name

    # sprite画像のパスを設定
    embedding.sprite.image_path = 'color_mnist_splite.png'
    embedding.sprite.single_image_dim.extend(28)

    # 設定を保存
    projector.visualize_embeddings(summary_writer, config)

    make_sprite(images, 'encoder/color_mnist_splite.png')
