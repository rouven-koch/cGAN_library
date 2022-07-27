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
        self.input_shape = (self.n_sites * self.n_omega)
        self.latent_dim = 10

        optimizer_gen = Adam(0.001, 0.5) #for generator and cGAN total
        optimizer_disc = 'SGD' #for discriminator

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer_disc) 

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['mean_squared_error'],
            optimizer=optimizer_gen) 
        
        
        #------ Until here ------
        
        
        
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.conditions,))
        inputs = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False


        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([inputs, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)