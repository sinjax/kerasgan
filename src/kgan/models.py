from keras.engine import Input, Model
from keras.layers import Dense, merge, Lambda
import keras.backend as K

import numpy as np

def d_loss(real_gen):
    real, gen = real_gen
    return K.mean(-K.log(real) - K.log(1 - gen),axis=1, keepdims=True)


def g_loss(gen):
    return K.mean(-K.log(gen),axis=1,keepdims=True)

class GAN(object):
    def __init__(self, discriminator, generator, d_opt="adam", g_opt="adam"):
        self.generator = generator
        self.discriminator = discriminator
        self.d_opt = d_opt
        self.g_opt = g_opt
        self._create_models()

    def _create_models(self):

        g_input = Input(self.generator.input_shape[1:])
        d_input = Input(self.discriminator.input_shape[1:])

        p_real = self.discriminator(d_input)
        p_gen = self.discriminator(self.generator(g_input))

        self.generator.trainable = False
        d_score = merge([p_real, p_gen], mode=d_loss, output_shape=(1,))
        self.d_trainer = Model(input=[d_input, g_input], output=d_score)
        self.d_trainer.compile(optimizer=self.d_opt, loss="mse")

        self.discriminator.trainable = False
        self.generator.trainable = True
        self.g_trainer = Model(input=[g_input], output=Lambda(g_loss)(p_gen))
        self.g_trainer.compile(optimizer=self.g_opt, loss="mse")

    def train_discriminator(self, x, y):
        return self.discriminator.train_on_batch(x,y)

    def train_gan(self, X, Z):
        diss_loss = self.d_trainer.train_on_batch([X, Z], [np.zeros((X.shape[0],1))])
        gen_loss = self.g_trainer.train_on_batch([Z], [np.zeros((X.shape[0],1))])

        return diss_loss, gen_loss
