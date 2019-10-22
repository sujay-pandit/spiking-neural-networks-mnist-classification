

####################################################### README ####################################################################
# This is the main file which calls all the functions and trains the network by updating weights
#####################################################################################################################################


import numpy as np
from neuron import neuron
import random
from matplotlib import pyplot as plt
from recep_field import rf
import imageio
from spike_train import encode
from rl import rl
from rl import update
from reconstruct import reconst_weights
from parameters import *
from var_th import threshold
import os
import pandas as pd
import time

pot_arrays = []
pth_arrays = []
for i in range(n):
	pot_arrays.append([])
	pth_arrays.append([])
#time series 
time_of_learning  = np.arange(1, T+1, 1)

output_layer = []

# creating the hidden layer of neurons
for i in range(n):
	a = neuron()
	a.initial()
	output_layer.append(a)

#Random synapse matrix	initialization
synapse = synapse_init
synapse_memory=np.zeros((n,m))

for k in range(epoch):
	print(k)
	for i in os.listdir("./train_mnist/"):

		img = imageio.imread("./train_mnist/"+i)

		#Convolving image with receptive field and encoding to generate spike train
		train = np.array(encode(rf(img)))

		#Local variables
		winner = False
		count_spikes= np.zeros(n)
		active_pot = np.zeros(n)

		#Leaky integrate and fire neuron dynamics
		for t in time_of_learning:
			for j, x in enumerate(output_layer):
				if(x.t_rest<t):
					x.P = x.P + np.dot(synapse[j], train[:,t])
					if(x.P>Prest):
						x.P -= var_D
					active_pot[j] = x.P
				
				pot_arrays[j].append(x.P) # Only for plotting: Changing potential overtime
				pth_arrays[j].append(x.Pth) # Only for plotting: Changing threshold overtime
			winner = np.argmax(active_pot)
			winner_synapses=[]

			#Check for spikes and update weights				
			for j,x in enumerate(output_layer):
				if(j==winner and active_pot[j]>output_layer[j].Pth):
					x.t_rest = t + x.t_ref
					x.P = Phyperpolarization
					x.Pth-= -1 ## Homoeostasis: Increasing the threshold of the neuron
					count_spikes[j]+=1
					for h in range(m):
						for t1 in range(0,t_back-1, -1): # if presynaptic spike came before postsynaptic spike
							if 0<=t+t1<T+1: 
								if train[h][t+t1] == 1: # if presynaptic spike was in the tolerance window
									synapse[j][h] = update(synapse[j][h], rl(t1)) # strengthen weights
									synapse_memory[j][h]=1
									break
						if synapse_memory[j][h]!=1: # if presynaptic spike was not in the tolerance window, reduce weights of that synapse
									synapse[j][h] = update(synapse[j][h], rl(3))
					for p in range(n):
						if p!=winner:
							if(output_layer[p].P>output_layer[p].Pth):
								count_spikes[p]+=1
							output_layer[p].inhibit()
					break
				
		# bring neuron potentials to rest
		for p in range(n):
				output_layer[p].initial()

		print("Image: "+i+" Spike COunt = ",count_spikes)
		print("Learning Neuron: ",np.argmax(count_spikes))
	
	# to write intermediate synapses for neurons
	for p in range(n):
			reconst_weights(synapse[p],str(p)+"_epoch_"+str(k))

np.savetxt("weights.csv", synapse, delimiter=",")

# Plotting
# ttt = np.arange(0,len(pot_arrays[0]),1)
# for i in range(n):
# 	axes = plt.gca()
# 	plt.plot(ttt,pth_arrays[i], 'r' )
# 	plt.plot(ttt,pot_arrays[i])
# 	plt.show()

#Reconstructing weights to analyse training
for i in range(n):
	reconst_weights(synapse[i],str(i)+"_final")