# coding: utf-8

import tensorflow as tf
import sonnet as snt


class ColorMNIST(snt.AbstractModule):
    def __init__(self, mnist, batch_size, name='ColorMNIST'):
        super(ColorMNIST, self).__init__(name=name)

        # データ数
        self._num_examples = mnist.num_examples
        # 画像(tf.constant)
        self._images = tf.constant(mnist.images, dtype=tf.float32)

        self._batch_size = batch_size

    def _build(self):
        # サンプラー(バッチ数)
        indices = tf.random_uniform([self._batch_size], 0, self._num_examples, tf.int64)
        x = tf.reshape(tf.gather(self._images, indices), (self._batch_size, 28, 28, 1))

        # 色を付与
        c = tf.random_normal([self._batch_size, 1, 1, 3], 0., 0.5)
        colored = x * c + (x - 1.)

        # ノイズを付与
        noise = tf.random_normal(colored.shape, 0, 0.2)
        noised = tf.clip_by_value(colored + noise, clip_value_max=1., clip_value_min=-1.)

        return noised
