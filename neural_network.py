# baseline cnn model for mnist
import numpy as np
from numpy import mean
from numpy import std
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler



class CNN():
    """ Class for building a Convolutional Neural Network and the
        application for many-body systems."""
        
    def __init__(self):
        # Inputs 
        self.pixel_x = 28 # number of pixel in x-direction
        self.pixel_y = 28 # number of pixel in y-direction
        self.input_shape = (self.pixel_x,self.pixel_y,1)
        
        # parameter for visualization
        self.xs = np.linspace(-10.0,10,self.pixel_x) 
        self.ys = np.linspace(-10.0,10,self.pixel_y) 

        # initialize CNN
        self.model = CNN_network()
        
        
    def CNN_network(self):
    """ Building the CNN network """       
        # build the layers
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])        
        # compile the model 
        model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        return model
    
    
    def training(self):






class NN():
    """ Parent class for Neural Networks"""
        
    def __init__(self, input_shape):
        print("--- Neural Network initialized ---")
        self.input_shape = input_shape
               
    def training(self, x_train, l_train, epochs=10, batch_size=128):
        return
       
    def predictions(self):
        return


class fcNN():
    """ Class for building a fully-connected Neural Network and the
        application for many-body systems."""
        
    def __init__(self):
        # Inputs 
        self.n_sites = 18 #number of sites of model
        self.n_omega = 50 #number of frequency points
        self.input_dim = self.n_sites * self.n_omega
        self.input_shape = (self.input_dim)
        
        # parameter for visualization
        self.xs = range(self.n_sites) #x-axis = sites in real space
        self.ys_spin = np.linspace(-0.0,2,self.n_omega) #energy (frequency) range with n_omega steps





# for plotting
nvalues = 28 #Number of values for each axis
xs = np.linspace(-10.0,10.0,nvalues)
ys = np.linspace(-10.0,10.0,nvalues)

# Plot spectrum (examples)
matplotlib.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure(dpi=150)
fig.subplots_adjust(0.2,0.2)
plt.contourf(xs,ys, spectra[5,:,:], 100)
plt.ylabel("muR = onsite right dot")
plt.xlabel("muL = onsite left dot")
plt.show()


# for plotting
nvalues = 28 #Number of values for each axis
xs = np.linspace(-10.0,10.0,nvalues)
ys = np.linspace(-10.0,10.0,nvalues)

# Plot spectrum (examples)
fig = plt.figure(dpi=150)
fig.subplots_adjust(0.2,0.2)
plt.contourf(xs,ys, spectra[9,:,:], 100)
plt.ylabel("muR = onsite right dot")
plt.xlabel("muL = onsite left dot")
plt.show()

#-----------------------------------------------------------------------------
# define model
def CNN_network():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.AUC(),])
    return model
#-----------------------------------------------------------------------------

model = CNN_network()

# training
batch_size = 64
epochs = 5

history = model.fit(trainX, l_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1)

# pedictions
pred_new = model.predict(testX)
mse_test = np.square(np.subtract(pred_new, l_test)).mean()
error_test = abs(np.subtract(pred_new, l_test)).mean()
error_individual = abs(np.subtract(pred_new, l_test))*100
print(error_test*100)

# plots
plot_loss(history)

# model.save('CNN_exp_delta.h5', model)



#-----------------------------------------------------------------------------
# reduce dimensions of 51x51 exp data

# (1) cut out important 28x28 part

data_cut = np.zeros((1,28,28,1))

data_exp_3D = np.zeros((51,51))

for j in range(51):
    for i in range(51):
        data_exp_3D[i,j]=data_exp[7,j*51+i]
 

# high resolution
muL_vals = np.linspace(-1, 1, 51) #On-site energies for left quantum dot
muR_vals = np.linspace(-1, 1, 51) #On-site energies for right quantum dot

mpl.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure(dpi=150)
fig.subplots_adjust(0.2,0.2)
plt.contourf(muL_vals,muR_vals, data_exp_3D, 100)
plt.ylabel("muR = onsite right dot")
plt.xlabel("muL = onsite left dot")
plt.show()

# low resolution
muL_vals = np.linspace(-1, 1, 28) #On-site energies for left quantum dot
muR_vals = np.linspace(-1, 1, 28) #On-site energies for right quantum dot

x_shift = 15 # bigger --> left
y_shift = 45 # bigger --> down

data_cut[0,:,:,0] = data_exp_3D[y_shift-28:y_shift,x_shift:28+x_shift]

mpl.rcParams['font.family'] = "Bitstream Vera Serif"
fig = plt.figure(dpi=150)
fig.subplots_adjust(0.2,0.2)
plt.contourf(muL_vals,muR_vals, data_cut[0,:,:,0], 100)
plt.vlines(0, -1, 1, lw=0.5)
plt.hlines(0, -1, 1, lw=0.5)
plt.ylabel("muR = onsite right dot")
plt.xlabel("muL = onsite left dot")
plt.show()

# scale data first!
data_cut_scale = data_cut * (1/np.amax(data_cut))

pred_exp = model.predict(data_cut_scale)
print(pred_exp)

pred_exp_scaled = scaler_alpha.inverse_transform(pred_exp)
print(pred_exp_scaled)
