# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import sonnet as snt
import os


class BaseModel(snt.AbstractModule):
    def __init__(self, name):
        super(BaseModel, self).__init__(name=name)

    def _build(self, **args):
        pass

    @property
    def variable_list(self):
        return [v for v in tf.global_variables() if v.name.startswith(self.module_name)]


class CodeDiscriminator(BaseModel):
    def __init__(self, name='Code_Discriminator', regularization=1.e-4):
        super(CodeDiscriminator, self).__init__(name=name)

        reg = {'w': l2_regularizer(scale=regularization),
               'b': l2_regularizer(scale=regularization)}

        with self._enter_variable_scope():
            self.l1 = snt.Linear(name='l1', output_size=750, regularizers=reg)
            self.bn1 = snt.BatchNorm(name='batch_norm_1')
            self.l2 = snt.Linear(name='l2', output_size=750, regularizers=reg)
            self.bn2 = snt.BatchNorm(name='batch_norm_2')
            self.l3 = snt.Linear(name='l3', output_size=1, regularizers=reg)

    def _build(self, z, is_train=False):
        h = tf.nn.relu(self.l1(z))
        h = self.bn1(h, is_train)
        h = tf.nn.relu(self.l2(h))
        h = self.bn2(h, is_train)
        h = self.l3(h)
        return tf.nn.sigmoid(h)


class AlphaGAN(object):
    def __init__(self, encoder, generator, discriminator, data, latent_size=10, lamb=1., max_iter=1000):
        self.e = encoder
        self.g = generator
        self.d = discriminator

        self.data = data

        self.c = CodeDiscriminator()

        self._latent_size = latent_size
        self._lamb = lamb
        self._max_iter = max_iter

        self._eps = 1.e-9

        self.x_real = None
        self.z_hat = None
        self.x_rec = None
        self.z = None
        self.x_gen = None

        # loss, optimizer, training step
        self.loss_e = self.optimizer_e = self.train_step_e = None
        self.loss_g = self.optimizer_g = self.train_step_g = None
        self.loss_d = self.optimizer_d = self.train_step_d = None
        self.loss_c = self.optimizer_c = self.train_step_c = None

    def _l1_loss(self, x_real, x_rec):
        with tf.name_scope('L1_loss'):
            loss = self._lamb * tf.reduce_sum(tf.abs(x_real - x_rec), axis=[1, 2, 3])
        return loss

    def _r(self, x):
        with tf.name_scope('R_x'):
            r = -tf.log(x + self._eps) + tf.log(1. - x + self._eps)
        return r

    def _loss_encoder(self, x_real, x_rec, cz_hat):
        with tf.name_scope('loss_encoder'):
            # L1 loss
            l1_loss = self._l1_loss(x_real, x_rec)

            # code loss
            code_loss = tf.squeeze(self._r(cz_hat))

            loss = tf.reduce_mean(l1_loss + code_loss)
        return loss

    def _loss_generator(self, x_real, x_rec, dx_rec, dx_gen):
        with tf.name_scope('loss_generator'):
            # L1 loss
            l1_loss = self._l1_loss(x_real, x_rec)

            # discriminator loss
            d_loss_real = self._r(dx_rec)
            d_loss_gen = self._r(dx_gen)

            loss = tf.reduce_mean(l1_loss + d_loss_real) + tf.reduce_mean(d_loss_gen)
        return loss

    def _loss_discriminator(self, dx_real, dx_rec, dx_gen):
        with tf.name_scope('loss_discriminator'):
            # loss real
            loss_real = -tf.log(dx_real + self._eps)

            # loss reconstruct
            loss_rec = - tf.log((1. - dx_rec) + self._eps)

            # loss generate
            loss_gen = -tf.log(1. - dx_gen + self._eps)

            loss = tf.reduce_mean(loss_real + loss_rec) + tf.reduce_mean(loss_gen)
        return loss

    def _loss_code_discriminator(self, cz, cz_hat):
        with tf.name_scope('loss_code_discriminator'):
            # loss real
            loss_real = -tf.log(1. - cz_hat + self._eps)

            # loss generate
            loss_gen = -tf.log(cz + self._eps)

            loss = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_gen)
        return loss

    def build_network(self):
        # get x
        self.x_real = self.data()

        # encode
        mu, s = self.e(self.x_real, True)
        with tf.name_scope('z_hat'):
            self.z_hat = mu + s * tf.random_normal(shape=mu.shape)

        # generate
        self.x_rec = self.g(self.z_hat, True)

        # sampling z
        with tf.name_scope('z'):
            self.z = tf.random_normal(shape=mu.shape)
        self.x_gen = self.g(self.z, True)

        # c(z)
        c_z_hat = self.c(self.z_hat, True)
        c_z = self.c(self.z, True)

        # d(x)
        d_x_real = self.d(self.x_real, True)
        d_x_rec = self.d(self.x_rec, True)
        d_x_gen = self.d(self.x_gen, True)

        # losses and optimizers
        self.loss_e = self._loss_encoder(x_real=self.x_real, x_rec=self.x_rec, cz_hat=c_z_hat)
        self.loss_g = self._loss_generator(x_real=self.x_real, x_rec=self.x_rec, dx_rec=d_x_rec, dx_gen=d_x_gen)
        self.loss_d = self._loss_discriminator(dx_real=d_x_real, dx_rec=d_x_rec, dx_gen=d_x_gen)
        self.loss_c = self._loss_code_discriminator(cz=c_z, cz_hat=c_z_hat)

        with tf.name_scope('Optimizer_Encoder'):
            self.optimizer_e = tf.train.AdamOptimizer(name='Adam_Encoder', learning_rate=1.e-4)
            self.train_step_e = self.optimizer_e.minimize(loss=self.loss_e, var_list=self.e.variable_list)

        with tf.name_scope('Optimizer_Generator'):
            self.optimizer_g = tf.train.AdamOptimizer(name='Adam_Generator', learning_rate=5.e-4)
            self.train_step_g = self.optimizer_g.minimize(loss=self.loss_g, var_list=self.g.variable_list)

        with tf.name_scope('Optimizer_Discriminator'):
            self.optimizer_d = tf.train.AdamOptimizer(name='Adam_Discriminator', learning_rate=5.e-4)
            self.train_step_d = self.optimizer_d.minimize(loss=self.loss_d, var_list=self.d.variable_list)

        with tf.name_scope('Optimizer_Code_Discriminator'):
            self.optimizer_c = tf.train.AdamOptimizer(name='Adam_Code_Discriminator', learning_rate=5.e-4)
            self.train_step_c = self.optimizer_c.minimize(loss=self.loss_c, var_list=self.c.variable_list)

    def set_summary(self):
        # set loss
        tf.summary.scalar('encoder_loss', self.loss_e)
        tf.summary.scalar('generator_loss', self.loss_g)
        tf.summary.scalar('discriminator_loss', self.loss_d)
        tf.summary.scalar('code_discriminator_loss', self.loss_c)

        # set images
        tf.summary.image('x_real', self.x_real, max_outputs=3)
        tf.summary.image('x_rec', self.x_rec, max_outputs=3)
        tf.summary.image('x_gen', self.x_gen, max_outputs=3)

    def train_network(self, log_dir):
        # build network
        self.build_network()

        # set summary
        self.set_summary()

        with tf.Session() as sess:
            # set writer and summary
            writer = tf.summary.FileWriter(log_dir, sess.graph)
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            # train
            for i in range(self._max_iter):
                # update encoder
                loss_e, _ = sess.run([alpha_gan.loss_e, alpha_gan.train_step_e])
                # update generator
                loss_g, _ = sess.run([alpha_gan.loss_g, alpha_gan.train_step_g])
                # update discriminator
                loss_d, _ = sess.run([alpha_gan.loss_d, alpha_gan.train_step_d])
                # update code discriminator
                loss_c, _ = sess.run([alpha_gan.loss_c, alpha_gan.train_step_c])

                summary = sess.run(merged)

                tf.logging.info('Step {}: Encoder loss={:.4f}, Generator loss={:.4f}, Discriminator loss={:.4f}, Code Discriminator loss={:.4f}'.format(i + 1, loss_e, loss_g, loss_d, loss_c))
                writer.add_summary(summary, i)

            # save
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(log_dir, 'alpha_gan_model.ckpt'))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    from mnist_vae import MNISTEncoder, MNISTGenerator, MNISTDiscriminator
    from color_mnist import ColorMNIST

    mnist = read_data_sets('./MNIST_data')

    mnist_encoder = MNISTEncoder(latent_size=50)
    mnist_generator = MNISTGenerator()
    mnist_discriminator = MNISTDiscriminator()

    color_mnist = ColorMNIST(mnist=mnist.train, batch_size=64)

    alpha_gan = AlphaGAN(encoder=mnist_encoder, generator=mnist_generator, discriminator=mnist_discriminator,
                         data=color_mnist, max_iter=100000)
    alpha_gan.train_network('./ColorMNIST')
