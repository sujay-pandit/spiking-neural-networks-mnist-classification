################################################ README #########################################################

# This file contains all the parameters of the network.

#################################################################################################################
import numpy as np
import random

# Simulation Parameters
T = 350 #Training time for every image
t_back = -5
epoch = 1


# Input Parameters
training_set_path= "./mnist_png/training/"
pixel_x = 28
Prest = -70
m = pixel_x*pixel_x # Number of neurons in first layer
n =  20  # Number of neurons in second layer

# Neuron Parameters
Pinhibit = -100 
Pth = -55
Phyperpolarization = -90
Pdrop = 0.8
Pthdrop = 0.4
synapse_init=np.zeros((n,m))
for i in range(n):
	for j in range(m):
		synapse_init[i][j] = random.uniform(0.95,1)
w_min=0.00001
w_max=np.max(synapse_init)

# STDP Parameters
sigma = 0.01 
A_plus = 0.8  
A_minus = 0.8 
tau_plus = 5
tau_minus = 5
mu=0.9


