"""
    @ file : gan_mnist.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2017.12.27
    @ version : 1.0
"""

import os, itertools, imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# reproducibility
seed = 200
np.random.seed(seed)
tf.set_random_seed(seed)


# Define functions
def generator(x):
    """
    Generator G(z)
    :param x: input vector (100-dim)
    :return: output vector (784-dim)
    """

    # initializer
    w_init = tf.contrib.layers.xavier_initializer() # tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)

    # number of hidden layers
    num_hidden1 = 256
    num_hidden2 = 512
    num_hidden3 = 1024
    num_output = 784    # MNIST dim (28x28)

    # 1st hidden layer
    w0 = tf.get_variable('w0', [x.get_shape()[1], num_hidden1], initializer=w_init)
    b0 = tf.get_variable('b0', [num_hidden1], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], num_hidden2], initializer=w_init)
    b1 = tf.get_variable('b1', [num_hidden2], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3nd hidden layer
    w2 = tf.get_variable('w2', [h1.get_shape()[1], num_hidden3], initializer=w_init)
    b2 = tf.get_variable('b2', [num_hidden3], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output layer
    w3 = tf.get_variable('w3', [h2.get_shape()[1], num_output], initializer=w_init)
    b3 = tf.get_variable('b3', [num_output], initializer=b_init)
    output = tf.nn.tanh(tf.matmul(h2, w3) + b3)

    return output


def discriminator(x, keep_prob):
    """
    Discriminator D(x)
    :param x: input vector (784-dim)
    :param keep_prob: dropout ratio
    :return: output vector (scalar)
    """

    # initializer
    w_init = tf.contrib.layers.xavier_initializer()  # tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)

    # number of hidden layers
    num_hidden1 = 512
    num_hidden2 = 512
    num_hidden3 = 1024
    num_output = 1

    # 1st hidden layer
    w0 = tf.get_variable('w0', [x.get_shape()[1], num_hidden1], initializer=w_init)
    b0 = tf.get_variable('b0', [num_hidden1], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, keep_prob=keep_prob)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], num_hidden2], initializer=w_init)
    b1 = tf.get_variable('b1', [num_hidden2], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

    # 3nd hidden layer
    w2 = tf.get_variable('w2', [h1.get_shape()[1], num_hidden3], initializer=w_init)
    b2 = tf.get_variable('b2', [num_hidden3], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, keep_prob=keep_prob)

    # output layer
    w3 = tf.get_variable('w3', [h2.get_shape()[1], num_output], initializer=w_init)
    b3 = tf.get_variable('b3', [num_output], initializer=b_init)
    output = tf.sigmoid(tf.matmul(h2, w3) + b3)

    return output


def optimizer(loss, var_list, learning_rate=0.0002):
    """
    Optimizer
    :param loss: loss function
    :param var_list: trainable variables
    :param learning_rate: learning rate
    :return: optimizer (ADAM)
    """

    step = tf.Variable(0, trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return opt


class GAN_Model(object):
    """ GAN_Model(sess, name, learning_rate)

        Build GAN graph
    """
    def __init__(self, sess, name, learning_rate=0.0002):
        """
        생성자
        :param sess: 세션
        :param name: 이름
        :param learning_rate: 학습 비율
        """
        self.sess = sess
        self.name = name

        # generator
        with tf.variable_scope('Generator'):
            self.z = tf.placeholder(tf.float32, shape=(None, 100))
            self.G = generator(self.z)

        # discriminator
        with tf.variable_scope('Discriminator') as scope:
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.x = tf.placeholder(tf.float32, shape=(None, 784))
            self.D1 = discriminator(self.x, self.keep_prob)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.keep_prob)

        # Define the loss for discriminator and generator
        eps = 0.0001  # to prevent log(0)
        self.loss_d = tf.reduce_mean(-tf.log(self.D1 + eps) - tf.log(1 - self.D2 + eps))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2 + eps))

        # trainable parameters
        self.params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        self.params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

        # optimizer
        self.opt_d = optimizer(self.loss_d, self.params_d, learning_rate)
        self.opt_g = optimizer(self.loss_g, self.params_g, learning_rate)

    def update_discriminator(self, x_data, z_data, keep_prob):
        """
        GAN 구별자 학습 (gradient descent 1 step)
        :param x_data: input data for D(x)
        :param z_data: noise data for G(z)
        :param keep_prob: dropout ratio
        :return: loss value of discriminator
        """
        loss_d, _ = self.sess.run([self.loss_d, self.opt_d],
                                  feed_dict={self.x: x_data, self.z: z_data, self.keep_prob: keep_prob})
        return loss_d

    def update_generator(self, z_data, keep_prob):
        """
        GAN 생성자 학습 (gradient descent 1 step)
        :param z_data: noise data for G(z)
        :param keep_prob: dropout ratio
        :return: loss value of generator
        """
        loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict={self.z: z_data, self.keep_prob: keep_prob})
        return loss_g


def run_train(model, display, train_set, training_epochs, batch_size):
    """
    GAN 학습 실행
    :param model: GAN model
    :param display: display tool
    :param train_set: training set
    :param training_epochs: number of epoch
    :param batch_size: size of batch data
    """

    print('\n===== Start : GAN training =====\n')

    # training-loop
    for epoch in range(training_epochs):

        avg_loss_d = 0
        avg_loss_g = 0
        total_batch = int(train_set.shape[0] / batch_size)

        for i in range(total_batch):

            # update discriminator
            x_data = train_set[i*batch_size:(i+1)*batch_size]
            z_data = np.random.normal(0, 1, (batch_size, 100))

            loss_d = model.update_discriminator(x_data, z_data, 0.7)
            avg_loss_d += loss_d / total_batch

            # update generator
            z_data = np.random.normal(0, 1, (batch_size, 100))

            loss_g = model.update_generator(z_data, 0.7)
            avg_loss_g += loss_g / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'loss_d =', '{:.9f}'.format(avg_loss_d),
              'loss_g =', '{:.9f}'.format(avg_loss_g))

        # show & save results
        p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'

        display.show_result((epoch + 1), save=True, path=p, is_fix=False)
        display.show_result((epoch + 1), save=True, path=fixed_p, is_fix=True)

    print('\n===== Finish : GAN training =====\n')


class Display(object):
    """ Display(sess, model)

        Display Tool
    """
    def __init__(self, sess, model):
        """
        생성자
        :param sess: session
        :param model: GAN model
        """
        self.sess = sess
        self.model = model
        self.fixed_z = np.random.normal(0, 1, (25, 100))

        # results save folder
        if not os.path.isdir('MNIST_GAN_results'):
            os.mkdir('MNIST_GAN_results')
        if not os.path.isdir('MNIST_GAN_results/Random_results'):
            os.mkdir('MNIST_GAN_results/Random_results')
        if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
            os.mkdir('MNIST_GAN_results/Fixed_results')

    def show_result(self, num_epoch, show=False, save=False, path='result.png', is_fix=False):
        """
        결과 그림 저장
        :param num_epoch:  current epoch
        :param show: TRUE or FALSE
        :param save: TRUE or FALSE
        :param path: save path
        :param is_fix: TRUE or FALSE
        """
        z_data = np.random.normal(0, 1, (25, 100))

        if is_fix:
            test_images = self.sess.run(self.model.G, feed_dict={self.model.z: self.fixed_z, self.model.keep_prob: 1.0})
        else:
            test_images = self.sess.run(self.model.G, feed_dict={self.model.z: z_data, self.model.keep_prob: 1.0})

        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5 * 5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()


def main():
    """
    Main function
    """

    # parameters
    learning_rate = 0.0002
    training_epochs = 100
    batch_size = 100

    # Load MNIST
    """ Check out https://www.tensorflow.org/get_started/mnist/beginners for
        more information about the mnist dataset 
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

    # initialization
    sess = tf.Session()
    gan = GAN_Model(sess=sess,
                    name='gan',
                    learning_rate=learning_rate,
                    )

    sess.run(tf.global_variables_initializer())

    # display option
    display = Display(sess=sess, model=gan)

    # training
    run_train(gan, display, train_set, training_epochs, batch_size)

    print("Training finish!... save training results")

    # Create animation
    images = []
    for e in range(training_epochs):
        img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

    sess.close()


if __name__ == '__main__':
    main()
