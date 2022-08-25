# cGAN library

from __future__ import print_function, division

# import libraries and packages
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from keras.layers import LeakyReLU, Activation, Input, Dense, Dropout, Concatenate, BatchNormalization, GaussianNoise
from keras.models import Model,Sequential
from keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, SGD
from keras import activations
from tensorflow.keras.models import load_model

from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate, UpSampling2D,MaxPool2D
from keras.layers import BatchNormalization, Activation, Embedding, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose

# check if GPU is used
tf.config.list_physical_devices('GPU')

# other  libraries
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler




class CGAN_CNN():
    """ Class for conditional generative adversarial networks (cGANS) 
        using Convolutional Neural Networks (CNNs)."""
        
    def __init__(self):
        # Inputs 
        self.n_sites = 18 # number of sites of model
        self.conditions = 2 # number of conditional parameters
        self.pixels = 28
        self.input_dim = self.pixels * self.pixels
        self.input_shape = (self.pixels, self.input_pixels)
        self.latent_dim = 100
        
        # parameter for visualization
        self.xs = np.linspace(-10.0,10.0,self.pixels) # x-axis ~ onsite energy 1
        self.ys = np.linspace(-10.0,10.0,self.pixels) # y-axis ~ onsite energy 2
        
        optimizer_gan = Adam(0.00001, 0.5) # optimizer for generator and cGAN total   
        
        self.regulizer = lambda: l1_l2(1e-5, 1e-5) #kenel-regulizer for both networks
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer_gan) 
           
        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['mean_squared_error'],
            optimizer=optimizer_gan) 
        
        # Build the cGAN
        self.discriminator.trainable = False
        self.cgan = self.build_cgan()
        self.cgan.compile(loss='binary_crossentropy', optimizer=optimizer_gan)
        
        
        
    def build_generator(self):
        # generator   
        z      = (Input(shape=(self.latent_dim,)))
        label  = (Input(shape=(self.conditions,)))
        inputs = (Concatenate())([z, label])       
        # layers
        g = (Dense(128*7*7, activation = 'relu'))(inputs)
        g = (Reshape((7,7,128)))(g)
        g = (Conv2DTranspose(256, (5,5), strides=(2, 2), padding='same', activation='relu'))(g)
        g = (Dropout(0.25))(g)
        g = (Conv2DTranspose(128, (5,5), strides=(2, 2), padding='same', activation='relu'))(g)
        g = (Dropout(0.25))(g)
        g = (Conv2D(1, (7,7), activation='sigmoid', padding='same'))(g)
        model = Model(inputs=[z, label], outputs=[g, label], name="generator")
        return model
    
    
    def build_discriminator(self):
        # discriminator    
        x      = (Input(shape=(self.pixels,self.pixels,1)))
        label  = (Input(shape=(self.conditions,)))    
        # layers
        d = (Conv2D(128, (5,5), padding='same', activation='relu', input_shape=(self.pixels,self.pixels,1)))(x)
        d = (MaxPool2D())(d)
        d = (Dropout(0.25))(d)
        d = (Conv2D(256, (5,5), strides=(2, 2), padding='same', activation='relu'))(d)
        d = (MaxPool2D(strides=(2,2)))(d)
        d = (Dropout(0.25))(d)
        d = (Flatten())(d)
        d = (Concatenate())([d, label])
        d = (Dense(128, activation='relu'))(d)
        d = (Dense(1))(d)
        d = (Activation('sigmoid'))(d)        
        model = Model(inputs=[x, label], outputs=d, name="discriminator")
        return model
        
    
    def build_cgan(self):
        # building the cGAN
        yfake = Activation("linear", name="yfake")(self.discriminator(self.generator(self.generator.inputs)))
        yreal = Activation("linear", name="yreal")(self.discriminator(self.discriminator.inputs))
        model = Model(self.generator.inputs + self.discriminator.inputs, [yfake, yreal], name="cGAN")
        return model



    
    def train(self, x_train, l_train, epochs=30, batch_size=64):
        " function for the training of the cGAN (can be quite challenging)"

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
                return None
         
    
    def save_weights_onsits(self, cGAN):
        "function to load trained model"
                
        # weights of the trained S=1 model
        self.generator.save_weights('generator_CNN_onsite.h5')
        self.discriminator.save_weights('discriminator_CNN_onsite.h5')
        self.cgan.save_weights('gan_CNN_onsite.h5')  
        
        return cGAN   
 
    
    def load_weights_onsits(self, cGAN):
        "function to load trained model"
                
        # weights of the trained S=1 model
        self.generator.load_weights('generator_CNN_onsite_exp.h5')
        self.discriminator.load_weights('discriminator_CNN_onsite_exp.h5')
        self.cgan.load_weights('gan_CNN_onsite_exp.h5')     
        
        return cGAN   
    
     
    # Update ----> 
        
    def format_inputs(self, data, label, val_split=0.1):
        "function to import and (pre) process dataset"
        
        # use MinMaxScaler for data and label
        scaler = MinMaxScaler(feature_range=(-1,1)) #data
        scaler_1 = MinMaxScaler(feature_range=(0,1)) #label 1
        scaler_2 = MinMaxScaler(feature_range=(0,1)) #label 2
        
        n_samples = len(data) #number of samples in the dataset
        
        # test and training data
        x_train = scaler.fit_transform(data[:int((1-val_split)*n_samples),:]) 
        x_test = scaler.transform(data[int((1-val_split)*n_samples):,:])
    
        # test and training labels            
        labels = label            
        l_train_1 = scaler_1.fit_transform(labels[:int((1-val_split)*n_samples),0:1]) 
        l_train_2 = scaler_2.fit_transform(labels[:int((1-val_split)*n_samples),1:]) 
        l_train = np.concatenate((l_train_1,l_train_2), axis=1)
    
        l_test_1 = scaler_1.transform(labels[int((1-val_split)*n_samples):,0:1]) 
        l_test_2 = scaler_2.transform(labels[int((1-val_split)*n_samples):,1:]) 
        l_test = np.concatenate((l_test_1,l_test_2), axis=1)
        
        return x_train, x_test, l_train, l_test, scaler, scaler_1, scaler_2
    
    
    def generate_G(self, param_1, param_2, scaler, scaler_1, scaler_2, batch_size=128):
        """ function to generate new samples.
            insert both conditional parameters, scaled in the interval I=[0,1]"""
        
        # initilize sampling 
        noise_batch = np.random.normal(0, 1, (batch_size, self.latent_dim)) 
        noise_batch[0] = np.random.uniform(0, 1, (1, self.latent_dim))
        generate_batch = np.zeros((batch_size,2))
        
        # conditional parameter
        generate_batch[0,0] = param_1 # insert here N_y scaled in [0.0, 1.0]  
        generate_batch[0,1] = param_2  # insert here B_y scaled in [0.0, 1.0] 
        
        # make new prediction
        pred = self.generator.predict([noise_batch, generate_batch])[0]
        pred_scale = scaler.inverse_transform(pred)
        
        # print real conditional parameter values
        param_1_rescaled = scaler_1.inverse_transform(generate_batch)
        print('conditional parameter (S=1 model):')
        print('N_y = ',  param_1_rescaled[0,0])
        param_2_rescaled = scaler_2.inverse_transform(generate_batch)
        print('B_x = ', param_2_rescaled[0,1])
                 
        # 3D plot
        plot_3D = np.zeros((self.n_omega, self.n_sites))
        for j in range(self.n_sites):
            for i in range(self.n_omega):
                plot_3D[i,j]=pred_scale[0,j*self.n_omega+i]
                
        matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
        fig = plt.figure()
        fig.subplots_adjust(0.2,0.2)
        plt.contourf(self.xs,self.ys_spin,plot_3D, 100)
        plt.ylabel("frequency [J]")
        plt.xlabel("site")
        plt.show()
        
        return None
    
    
    
    
    # next functions to implement and test:

        
    def data_2D_to_3D(self, data_2D):
        """ convert 2D (input shape) to 3D data (plot shape)"""        
        data_3D = np.zeros((len(data_2D), self.n_omega, self.n_sites))
        for k in range(len(data_2D)):
            for j in range(self.n_sites):
               for i in range(self.n_omega):
                   data_3D[i,j]=data_2D[k,j*self.n_omega+i]                        
        return data_3D
    
    
    def data_3D_to_2D(self, data_3D):
        """ convert 3D (plot shape) 2D data (input shape)"""       
        n_data = len(data_3D)
        data_2D = np.zeros((n_data, self.n_omega*self.n_sites))        
        for k in range(n_data):
            for m in range(self.n_sites):
                for l in range(self.n_omega):
                    data_2D[k,m*self.n_omega+l] = data_3D[k,l,m]                     
        return data_2D
    
        
    def save_weights(self):
        "function to save weights"
        return None
        
    
    def create_3D_plot(self):
        """ create 3D plot of dynamical correlator """
        return None

    
    def param_estimation(self):
        "estimate Hamiltonian parameters"
        return None
    
    
    
    
# # TEST CODE HERE    
    
# if __name__ == '__main__':
    
#     # define cGAN
#     cgan = CGAN()
    
#     # show architecture
#     # cgan.cgan.summary()
#     cgan.generator.summary()
#     cgan.discriminator.summary()
    
#     # full data
#     data_all = np.load("/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/S1/x_data_all.npy")
#     label_all = np.load("/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/S1/label_all.npy")
    
#     # preprocess data 
#     x_train, x_test, l_train, l_test, scaler, scaler_1, scaler_2 = cgan.format_inputs(data_all, label_all)
    
#     # load weights of pretrained model (here: only for the S=1 model)
#     cgan = cgan.load_weights_S1(cgan)

#     # generate new sample
#     cgan.generate_S1(0.5, 0.5, scaler, scaler_1, scaler_2)
    
#     # parameter estimation (2 parameters)

#     # test training
#     #cgan.train(x_train,l_train)