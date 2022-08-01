# cGAN library

from __future__ import print_function, division

# ML libraries
import tensorflow as tf
from keras import activations
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD

# other  libraries
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler




class CGAN():
    """ Class for conditional generative adversarial networks (cGANS) and the
        applications for many-body systems."""
        
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
        
        
    def format_inputs(self, data, label, val_split=0.1):
        "function to import and (pre) process dataset"
        
        # use MinMaxScaler for data and label
        scaler = MinMaxScaler(feature_range=(-1,1)) #data
        scaler_alpha = MinMaxScaler(feature_range=(0,1)) #label 1
        scaler_beta = MinMaxScaler(feature_range=(0,1)) #label 2
        
        n_samples = len(data) #number of samples in the dataset
        
        # test and training data
        x_train = scaler.fit_transform(data[:int((1-val_split)*n_samples),:]) 
        x_test = scaler.transform(data[int((1-val_split)*n_samples):,:])
    
        # test and training labels            
        labels = label            
        l_train_1 = scaler_alpha.fit_transform(labels[:int((1-val_split)*n_samples),0:1]) 
        l_train_2 = scaler_beta.fit_transform(labels[:int((1-val_split)*n_samples),1:]) 
        l_train = np.concatenate((l_train_1,l_train_2), axis=1)
    
        l_test_1 = scaler_alpha.transform(labels[int((1-val_split)*n_samples):,0:1]) 
        l_test_2 = scaler_beta.transform(labels[int((1-val_split)*n_samples):,1:]) 
        l_test = np.concatenate((l_test_1,l_test_2), axis=1)
        
        return x_train, x_test, l_train, l_test
        
        
    # next functions to implement:

    def load_weights(self, cGAN):
        "function to load trained model"
                
        # weights of the trained S=1 model
        self.generator.load_weights('/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/code/generator_CcGAN_MB_18_S1.h5')
        self.discriminator.load_weights('/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/code/discriminator_CcGAN_MB_18_S1.h5')
        self.cgan.load_weights('/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/code/gan_CcGAN_MB_18_S1.h5')

        return cGAN
    
    
    def save_weights(self):
        "function to save weights"
        return None
    
    
    def cgan_generate(self):
        "generate new samples"
        
        # initilize sampling 
        noise_batch = np.random.normal(0, 1, (batch_size, noise_dim)) 
        noise_batch[0] = np.random.uniform(0, 1, (1, noise_dim))
        test_label_batch = np.zeros((batch_size,2))
        
        # conditional parameter
        test_label_batch[0,0] = 0.3 # insert here N_y scaled in [0.0, 1.0]  
        test_label_batch[0,1] = 0.9  # insert here B_y scaled in [0.0, 1.0] 
        
        # make new prediction
        pred = generator_S1.predict([noise_batch, test_label_batch])[0]
        pred_scale = scaler.inverse_transform(pred)
        
        # print real conditional parameter values
        N_real = scaler_alpha.inverse_transform(test_label_batch)
        print('conditional parameter:')
        print('N_y =',  N_real[0,0])
        B_real = scaler_beta.inverse_transform(test_label_batch)
        print('B_x =', B_real[0,1])
        
        # 3D plot
        d3_plot_gan = np.zeros((omega_dim,18))
        for j in range(18):
            for i in range(omega_dim):
                d3_plot_gan[i,j]=pred_scale[0,j*omega_dim+i]
        
        matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
        fig = plt.figure()
        fig.subplots_adjust(0.2,0.2)
        plt.contourf(xs,ys_spin,d3_plot_gan,100)
        plt.ylabel("frequency [J]")
        plt.xlabel("Site")
        plt.show()
        
        # plot validation data
        n_plot = 0
        print(label_test_1[n_plot])
        
        matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
        fig = plt.figure()
        fig.subplots_adjust(0.2,0.2)
        plt.contourf(xs,ys_spin,dos_test_1[n_plot,:,:],100)
        plt.ylabel("frequency [J]")
        plt.xlabel("Site")
        plt.show()
        
        return None
    
    
    def cgan_param_estimation(self):
        "estimate Hamiltonian parameters"
        return None
    
    
    
if __name__ == '__main__':
    
    # define cGAN
    cgan = CGAN()
    
    # show architecture
    # cgan.cgan.summary()
    cgan.generator.summary()
    cgan.discriminator.summary()
    
    # full data
    data_all = np.load("/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/S1/x_data_all.npy")
    label_all = np.load("/u/11/kochr1/unix/Rouven/Python/Project_2/results_paper/S1/label_all.npy")
    
    # preprocess data 
    x_train, x_test, l_train, l_test = cgan.format_inputs(data_all, label_all)
    
    # load weights of pretrained model (here: only for the S=1 model)
    cgan = cgan.load_weights(cgan)

    # generate new sample
    
    # parameter estimation (2 parameters)

    # test training
    #cgan.train(x_train,l_train)