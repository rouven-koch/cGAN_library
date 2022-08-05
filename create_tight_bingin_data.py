# create tight-binding systems as training data for the cGAN

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib

# create Hamiltonian
n = 18 #dimension of 1d chain
h = np.zeros((n,n))

def hamiltonian_1d(n):   
    # random variables
    mu_random = np.random.uniform(-0.3, 0.3) #variation in onsite energies
    delta =  np.random.uniform(-0.3, 0.3) #site imbalance 
    noise_magn =  np.random.uniform(0.0, 1.0) #magnitude of noise
    w_on_all = np.zeros(n)
    
    for i in range(n):
        # diagonal elements
        w_on = noise_magn * np.random.uniform(-0.50, 0.50)
        w_on_all[i] = w_on
        t_hop = 1 #hopping term
        mu = 2  
        mu = mu + mu_random
        h[i,i] = w_on + (-1)**i*delta + mu
        
    for j in range(n-1):
        # offdiagonal elements
        h[j+1,j] = t_hop
        h[j,j+1] = t_hop
        
    return h, delta, mu_random, noise_magn 


# compute local DOS
def local_dos(h, omega, n, site):
    epsilon = 0.1
    c_1 = 0.010 # + np.random.uniform(0.005, 0.02)
    c_2 = 0.005 # + np.random.uniform(0.005, 0.02)
    epsilon += c_1*omega + c_2*(omega**2)
    m = h - np.identity(n)*(omega-1j*epsilon)
    m_inverse = linalg.inv(m)
    m_complex = m_inverse.imag
    return m_complex[site,site], c_1, c_2


# create data
n_data = 1
n = 18 #dimension of 1d chain
labels = np.zeros((n_data,3+2+n)) # last label = noise // for training of extra ANN later
h_all = np.zeros((n,n,n_data))

# save hamiltonians and labels
for i in range(n_data):    
    # h_all[:,:,i], labels[i,0], labels[i,1], labels[i,2:20] = hamiltonian_1d(n)
    h_all[:,:,i], labels[i,0], labels[i,1], labels[i,2] = hamiltonian_1d(n)
    
# create D(omega,n)
omega_values = np.arange(0, 2.5, 0.05)
dos_all = np.zeros((n_data, len(omega_values), n))
dos_all_format = np.zeros((n_data, len(omega_values)*n))

# format DOS
for k in range(n_data):
    for count, omega in enumerate(omega_values): 
        for i in range(n):
            dos_all[k,count,i], labels[k,3], labels[k,4] = local_dos(h_all[:,:,k], omega=omega, n=n, site=i)  
            dos_all_format[k,count*n+i], labels[k,n+2], labels[k,n+3] = local_dos(h_all[:,:,k], omega=omega, n=n, site=i)  

# save dos_all and labels as input for CcGAN
np.save("dos.npy", dos_all) 
np.save("dos_format.npy", dos_all_format) 
np.save("labels.npy", labels) 







