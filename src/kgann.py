import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.initializations import normal
from keras.layers import Dense
from scipy.stats import norm

from kgan.models import GAN

np.random.seed(42)


class DataDistribution(object):
    def __init__(self):
        self.mu = -4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
               np.random.random(N) * 0.01


def d_loss(real_gen):
    real, gen = real_gen
    return K.mean(-K.log(real) - K.log(1 - gen), axis=1, keepdims=True)


def g_loss(gen):
    return K.mean(-K.log(gen), axis=1, keepdims=True)


def tf_norm(shape, name=None):
    return normal(shape, name=name, scale=0.1)


class KerasGAN(object):
    def __init__(self):
        self.f, self.ax = plt.subplots(1)
        self.nsteps = 2200
        self.batch_size = 128
        self.data_distribution = DataDistribution()
        self.generator_distribution = GeneratorDistribution(range=8)
        self.num_pretrain_steps = 1000
        self._create_models()

    def _create_models(self, nh=8):
        g_input = Input((1,))
        h = Dense(nh, init=tf_norm, input_shape=(1,))(g_input)
        gen = Dense(1, init=tf_norm)(h)
        generator = Model(g_input, gen)
        generator.compile(optimizer="adam", loss="mse")

        d_input = Input((1,))
        H = Dense(nh * 2, init=tf_norm, input_shape=(1,), activation="tanh")(d_input)
        H = Dense(nh * 2, init=tf_norm, activation="tanh")(H)
        H = Dense(nh * 2, init=tf_norm, activation="tanh")(H)
        d_out = Dense(1, init=tf_norm, activation="sigmoid")(H)
        discriminator = Model(input=[d_input], output=d_out)
        discriminator.compile(optimizer="adam", loss="mse")

        self.gan = GAN(discriminator, generator)

    def samples(self, num_points=10000, num_bins=100):
        """
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        """
        xs = np.linspace(-self.generator_distribution.range, self.generator_distribution.range, num_points).reshape(
            (num_points, 1))
        bins = np.linspace(-self.generator_distribution.range, self.generator_distribution.range, num_bins)

        # decision boundary
        db = self.gan.discriminator.predict([xs])

        # data distribution
        d = self.data_distribution.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        g = self.gan.generator.predict([xs])
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def plot_distributions(self, block=False):
        db, pd, pg = self.samples()
        db_x = np.linspace(-self.generator_distribution.range, self.generator_distribution.range, len(db))
        p_x = np.linspace(-self.generator_distribution.range, self.generator_distribution.range, len(pd))
        self.ax.clear()
        self.ax.plot(db_x, db, label='decision boundary')
        self.ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show(block=block)
        plt.savefig("last_fig.png")
        plt.pause(0.05)

    def train(self):
        print("Pre Training...")
        for step in range(self.num_pretrain_steps):
            d = (np.random.random(self.batch_size) - 0.5) * 10.0
            labels = norm.pdf(d, loc=self.data_distribution.mu, scale=self.data_distribution.sigma).reshape(
                (self.batch_size, 1))
            self.gan.train_discriminator([d.reshape((self.batch_size, 1))], [labels])

        print("Training")
        for step in range(self.nsteps):
            X = self.data_distribution.sample(self.batch_size).reshape((self.batch_size, 1))
            Z = self.generator_distribution.sample(self.batch_size).reshape((self.batch_size, 1))

            diss_loss, gen_loss = self.gan.train_gan(X, Z)

            if step % 10 == 0:
                print("Step: %d" % step)
                self.plot_distributions()
                print("Loss: dis: %s, gen: %s" % (diss_loss, gen_loss))
        self.plot_distributions(True)


gan = KerasGAN()
gan.train()
