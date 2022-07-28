# cGAN library

from __future__ import print_function, division

import tensorflow as tf
from keras import activations
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import numpy as np

import sys

import seaborn as sns



class CGAN():
    
    def __init__(self):
        # Inputs 
        self.n_sites = 18 #number of sites of model
        self.n_omega = 50 #number of frequency points
        self.conditions = 2 #number of conditional parameters
        self.input_dim = self.n_sites * self.n_omega
        self.input_shape = (self.input_dim)
        self.latent_dim = 10

        optimizer_gen = Adam(0.001, 0.5) #optimizer for generator and cGAN total
        optimizer_disc = 'SGD' #optimizer for discriminator
        
        self.regulizer = lambda: l1_l2(1e-5, 1e-5) #kenel-regulizer for both networks
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer_disc) 
           
        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['mean_squared_error'],
            optimizer=optimizer_gen) 
        
        # Build the cGAN
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.cgan = self.build_cgan()
        self.cgan.compile(loss='binary_crossentropy', optimizer=optimizer_gen)
        
        
    def build_generator(self):
        # generator
        z      = Input(shape=(self.latent_dim,))
        label  = Input(shape=(self.conditions,))
        inputs = (Concatenate())([z, label])
        g = (Dense(int(4048 / 2), kernel_regularizer=self.regulizer()))(inputs)
        g = (LeakyReLU(0.2))(g)
        g = (Dense(int(4048), kernel_regularizer=self.regulizer()))(g)
        g = (LeakyReLU(0.2))(g)
        g = (Dense(np.prod(self.input_dim), kernel_regularizer=self.regulizer()))(g)
        g = (Activation(activations.tanh))(g)    
        model = Model(inputs=[z, label], outputs=[g, label], name="generator")
        return model
    
    def build_discriminator(self):
        # discriminator
        x      = Input(shape=(self.input_dim,))
        label  = Input(shape=(self.conditions,))
        inputs = (Concatenate()([x, label]))
        d = (GaussianNoise(0.033))(inputs)
        d = (Dense(4048, kernel_regularizer=self.regulizer()))(inputs)
        d = (LeakyReLU(0.2))(d)
        d = (Dense(int(4048 / 2), kernel_regularizer=self.regulizer()))(d)
        d = (LeakyReLU(0.2))(d)
        d = (Dense(1, kernel_regularizer=self.regulizer()))(d)
        d = (Activation('sigmoid'))(d)
        model = Model(inputs=[x, label], outputs=d, name="discriminator")
        return model
        
    def build_cgan(self):
        # building the cGAN
        yfake = Activation("linear", name="yfake")(self.discriminator(self.generator(self.generator.inputs)))
        yreal = Activation("linear", name="yreal")(self.discriminator(self.discriminator.inputs))
        model = Model(self.generator.inputs + self.discriminator.inputs, [yfake, yreal], name="cGAN")
        return model
    
    def train(self,x_train, l_train, epochs=10, batch_size=128):
        # training parameter
        num_batches = int(x_train.shape[0] / batch_size)
        for epoch in range(epochs):            
            print("Epoch {}/{}".format(epoch + 1, epochs))
            for index in range(num_batches):                
                # train discriminator
                self.discriminator.trainable = True
            
                # train discriminator on real data
                batch       = np.random.randint(0, x_train.shape[0], size=batch_size)
                input_batch = x_train[batch]
                label_batch = l_train[batch]
                y_real      = np.ones(batch_size) + 0.1 * np.random.uniform(-1, 1, size=batch_size)
                self.discriminator.train_on_batch([input_batch, label_batch], y_real)
                
                # train discriminator on fake data
                noise_batch      = np.random.normal(0, 1, (batch_size, self.latent_dim))    
                generated_images = self.generator.predict([noise_batch, label_batch])
                y_fake           = np.zeros(batch_size) + 0.1 * np.random.uniform(0, 1, size=batch_size)
                d_loss = self.discriminator.train_on_batch(generated_images, y_fake)  
                self.discriminator.trainable = False
                
                # train GAN
                gan_loss = self.cgan.train_on_batch([noise_batch, label_batch, input_batch, label_batch], [y_real, y_fake])
                print("Batch {}/{}: Discriminator loss = {}, GAN loss = {}".format(index + 1, num_batches, d_loss, gan_loss))
        
            
    
if __name__ == '__main__':
    cgan = CGAN()
    # show architecture --> compare with old one [SAME :)]
    # cgan.cgan.summary()
    cgan.generator.summary()
    cgan.discriminator.summary()
    
    # test training
    cgan.train(x_train,l_train_scale)