################################################ README #########################################################

# This file contains all the parameters of the network.

#################################################################################################################
import numpy as np
import random
scale = 1
T = 350 #(us) Training time for every image
t_back = -5 #(us)
t_fore = 5 #(us)

pixel_x = 28
Prest = -70 #(mV)
m = pixel_x*pixel_x #Number of neurons in first layer
n =  3  #Number of neurons in second layer
Pinhibit = -200 #(mV) 
Pth = -55 #(mV)
Phyperpolarization = -90
var_D = 0.1

synapse_init=np.zeros((n,m))
for i in range(n):
	for j in range(m):
		synapse_init[i][j] = random.uniform(0,0.3)
w_min=np.min(synapse_init)
w_max=np.max(synapse_init)
# w_max = 1
# w_min = 0
sigma = 0.01 #0.02
A_plus = 0.8  # time difference is positive i.e negative reinforcement
A_minus = 0.8 # 0.01 # time difference is negative i.e positive reinforcement 
tau_plus = 100
tau_minus = 100

epoch = 1

mu=0.9
fr_bits = 12
int_bits = 12