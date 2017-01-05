import json
import random

from keras.datasets import mnist
from keras.engine import Input, Model
from keras.layers import Dense, BatchNormalization, Activation, Reshape, UpSampling2D, Convolution2D, LeakyReLU, \
    Dropout, \
    Flatten

import numpy as np
from keras.models import load_model
from keras.optimizers import Adam, Optimizer
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from kgan.models import GAN
import os
import pickle


def inputExplorer(f, sliders_properties, wait_for_validation=False):
    """ A light GUI to manually explore and tune the outputs of
        a function.
        slider_properties is a list of dicts (arguments for Slider )
        whose keys are in ( label, valmin, valmax, valinit=0.5,
        valfmt='%1.2f', closedmin=True, closedmax=True, slidermin=None,
        slidermax=None, dragging=True)

        def volume(x,y,z):
            return x*y*z

        intervals = [ { 'label' :  'width',  'valmin': 1 , 'valmax': 5 },
                  { 'label' :  'height',  'valmin': 1 , 'valmax': 5 },
                  { 'label' :  'depth',  'valmin': 1 , 'valmax': 5 } ]
        inputExplorer(volume,intervals)
    """

    nVars = len(sliders_properties)
    slider_width = 1.0 / nVars
    print(slider_width)

    # CREATE THE CANVAS

    figure, ax = plt.subplots(1)
    figure.canvas.set_window_title("Inputs for '%s'" % f)

    # choose an appropriate height

    width, height = figure.get_size_inches()
    height = min(0.5 * nVars, 8)
    figure.set_size_inches(width, height, forward=True)

    # hide the axis
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # CREATE THE SLIDERS

    sliders = []

    for i, properties in enumerate(sliders_properties):
        ax = plt.axes([0.1, 0.95 - 0.9 * (i + 1) * slider_width,
                       0.8, 0.8 * slider_width])
        sliders.append(Slider(ax=ax, **properties))

    # CREATE THE CALLBACK FUNCTIONS

    def on_changed(event):
        res = f(*(s.val for s in sliders))
        if res is not None:
            print(res)

    def on_key_press(event):
        if event.key is 'enter':
            on_changed(event)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    # AUTOMATIC UPDATE ?
    if not wait_for_validation:
        for s in sliders:
            s.on_changed(on_changed)

    # DISPLAY THE SLIDERS

    plt.show()


def points(a, b, n=100):
    diff = (b - a) / n
    ret = [a]
    while len(ret) < n and ((a < b and (a + diff) < b) or (a > b and (a + diff) > b)):
        a += diff
        ret += [a]
    if len(ret) != n:
        ret += [ret[-1] for _ in range(n - len(ret))]
    return ret


class MNISTGan(object):
    def __init__(self, output, generator=None, discriminator=None, g_adam=None, d_adam=None, step=None, input_sample_size=10, pretrain=False):
        self.figure = None
        self.output = output
        dropout_rate = 0.25
        self.batch_size = 32
        self.nsteps = 5000
        self.input_sample_size = input_sample_size
        self.step = step
        self.pretrain = pretrain

        first_save = generator is None or discriminator is None

        self.load_dataset()

        if g_adam is None:
            self.g_adam = Adam(lr=1e-03)
        else:
            self.g_adam = g_adam

        if d_adam is None:
            self.d_adam = Adam(lr=1e-03)
        else:
            self.d_adam = d_adam

        nch = 200
        if generator is None:
            g_input = Input(shape=[self.input_sample_size])
            H = Dense(nch * (self.img_rows // 2) * (self.img_cols // 2), init='glorot_normal')(g_input)
            H = BatchNormalization(mode=0)(H)
            H = Activation('relu')(H)
            H = Reshape([nch, self.img_rows // 2, self.img_cols // 2])(H)
            H = UpSampling2D(size=(2, 2))(H)
            H = Convolution2D(nch // 2, 3, 3, border_mode='same', init='glorot_uniform')(H)
            H = BatchNormalization(mode=0)(H)
            H = Activation('relu')(H)
            H = Convolution2D(nch // 4, 3, 3, border_mode='same', init='glorot_uniform')(H)
            H = BatchNormalization(mode=0)(H)
            H = Activation('relu')(H)
            H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
            g_V = Activation('sigmoid')(H)
            generator = Model(g_input, g_V)
        generator.compile(loss='binary_crossentropy', optimizer="adam")
        generator.summary()

        shp = self.X_train.shape[1:]

        if discriminator is None:
            d_input = Input(shape=shp)
            H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(d_input)
            H = LeakyReLU(0.2)(H)
            H = Dropout(dropout_rate)(H)
            H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(H)
            H = LeakyReLU(0.2)(H)
            H = Dropout(dropout_rate)(H)
            H = Flatten()(H)
            H = Dense(256)(H)
            H = LeakyReLU(0.2)(H)
            H = Dropout(dropout_rate)(H)
            d_V = Dense(1, activation='sigmoid')(H)
            discriminator = Model(d_input, d_V)
        discriminator.compile(loss='binary_crossentropy', optimizer="adam")
        discriminator.summary()

        self.gan = GAN(discriminator, generator, d_opt=self.d_adam, g_opt=self.g_adam)
        if first_save:
            self._save_model()

    def load_dataset(self):
        self.img_rows, self.img_cols = 28, 28
        # the data, shuffled and split between train and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.img_rows, self.img_cols)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.img_rows, self.img_cols)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

    def train(self):
        if self.step is None:
            self.step = 0

            if self.pretrain:
                print("Pretraining discriminator")
                ntrain = 1000
                trainidx = random.sample(range(0, self.X_train.shape[0]), ntrain)
                XT = self.X_train[trainidx, :, :, :]
                print("... generating noise")
                noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], self.input_sample_size])
                generated_images = self.gan.generator.predict(noise_gen)
                X = np.concatenate((XT, generated_images))
                n = XT.shape[0]
                y = np.zeros([2 * n, 1])
                y[:n, 0] = 1
                y[n:, 0] = 0
                self.gan.discriminator.fit(X, y, nb_epoch=1, batch_size=128)
        else:
            self.plot_generator()
            self.step += 1
            print("Starting from step: %d" % self.step)

        for step in range(self.step, self.nsteps):
            self.step = step
            print("Starting step: %d" % step)
            X = self.X_train[np.random.randint(0, self.X_train.shape[0], size=self.batch_size), :, :, :]
            Z = np.random.uniform(0, 1, size=[self.batch_size, self.input_sample_size])
            d_loss, g_loss = self.gan.train_gan(X, Z)
            print("Loss: D(%s), G(%s)" % (d_loss, g_loss))
            if step % 10 == 0:
                print("Saving model to: %s" % self.output)
                self._save_model()
                self.plot_generator()

    def play(self):
        def do_something(*vals):
            arr = np.array([vals])
            self.plot_generator(arr)

        init_vals = np.random.uniform(0, 1, size=[self.input_sample_size]).tolist()
        do_something(*init_vals)
        inputExplorer(do_something, [
            {"label": "v_%d" % i, "valmin": 0, "valmax": 1, "valinit": init_vals[i]}
            for i in range(self.input_sample_size)
        ])

    def plot_generator(self, noise=None, n_ex=16, dim=(4, 4), figsize=(10, 10)):
        if noise is None:
            noise = np.random.uniform(0, 1, size=[n_ex, self.input_sample_size])
        else:
            n_ex = noise.shape[0]
            dim = (int(np.sqrt(n_ex)), np.ceil(n_ex / int(np.sqrt(n_ex))))
        generated_images = self.gan.generator.predict(noise)
        if self.figure is None:
            self.figure = plt.figure(figsize=figsize)

        self.figure.clear()
        # self.figure.clear()
        plt.figure(self.figure.number)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            img = generated_images[i, 0, :, :]
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

    def animate_gen(self, steps=100):
        animation_path = np.array(
            [points(random.uniform(0, 1), random.uniform(0, 1), steps) for _ in range(self.input_sample_size)]
        ).T
        ret = self.gan.generator.predict(
            np.vstack((animation_path, np.flipud(animation_path)))
        )
        imagelist = [ret[i, 0, :, :] for i in range(ret.shape[0])]
        fig = plt.figure()  # make figure

        im = plt.imshow(imagelist[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=1)

        # function to update figure
        def updatefig(j):
            # set the data in the axesimage object
            im.set_array(imagelist[j])
            # return the artists set
            return im,

        # kick off the animation
        ani = animation.FuncAnimation(fig, updatefig, frames=range(steps * 2),
                                      interval=50)
        plt.show()

    def _save_model(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.gan.generator.save("%s/generator.keras" % self.output)
        self.gan.discriminator.save("%s/discriminator.keras" % self.output)
        json.dump({"steps": self.step, "input_sample_size": self.input_sample_size}, open("%s/stats.json" % self.output, "w"))

    @staticmethod
    def load_model(output):
        generator = load_model("%s/generator.keras" % output)
        discriminator = load_model("%s/discriminator.keras" % output)

        stats = json.load(open("%s/stats.json" % output))
        mnist_gan = MNISTGan(
            output, generator, discriminator,
            step=stats["steps"],
            input_sample_size=stats["input_sample_size"]
        )
        return mnist_gan


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-output", "-o", "--output", default="data/mnist_gan")
    sp = ap.add_subparsers(help="mode")
    play_sp = sp.add_parser("play")
    play_sp.set_defaults(func=lambda net: net.play())
    train_sp = sp.add_parser("train")
    train_sp.set_defaults(func=lambda net: net.train())
    train_sp.add_argument("-input-sample-size", "--input-sample-size", "-is", default=100, type=int)
    animate_sp = sp.add_parser("animate")
    animate_sp.set_defaults(func=lambda net: net.animate_gen())
    parsed = ap.parse_args()
    if os.path.exists(parsed.output):
        print("Loading existing GAN from: %s" % parsed.output)
        net = MNISTGan.load_model(parsed.output)
    else:
        print("Creating new MNISTGan")
        net = MNISTGan(parsed.output, input_sample_size=parsed.input_sample_size)

    parsed.func(net)
