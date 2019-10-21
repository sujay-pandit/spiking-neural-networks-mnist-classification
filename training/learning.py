

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
#potentials of output neurons

### TO CREATE "EXPERIENCE" FOR NEURONS
pot_arrays = []
for i in range(n):
	pot_arrays.append([])
#time series 
time_of_learning  = np.arange(1, T+1, 1)

layer2 = []

# creating the hidden layer of neurons
for i in range(n):
	a = neuron()
	a.initial()
	layer2.append(a)

#synapse matrix	initialization
synapse = synapse_init

synapse_memory=np.zeros((n,m))

for k in range(epoch):
	print(k)
	for i in os.listdir("./train_mnist/"):

		img = imageio.imread("./train_mnist/"+i)

		## Some synaptic flags
		tmp_synapse =  synapse_init
		tmp_synapse_memory=np.zeros((n,m))
		#Convolving image with receptive field
		pot = rf(img)
		# imageio.imwrite(str(i)+".png",pot)
		#Generating spike train
		train = np.array(encode(pot))

		#calculating threshold value for the image
		#var_threshold = threshold(train)
		
	

		# for x in layer2:
		# 	x.initial()

		#flag for lateral inhibition
		winner = False
		count_wins= np.zeros(n)
		active_pot = np.zeros(n)

		#Leaky integrate and fire neuron dynamics
		for t in time_of_learning:
			for j, x in enumerate(layer2):
				if(x.t_rest<t):
					x.P = x.P + np.dot(synapse[j], train[:,t])
					if(x.P>Prest):
						x.P -= var_D
					active_pot[j] = x.P
				
				pot_arrays[j].append(x.P) ## Only for plotting: Changing potential overtime
			winner = np.argmax(active_pot)
			winner_synapses=[]

			#Check for spikes and update weights				
			for j,x in enumerate(layer2):
				#s = x.check() # Check for SPike; if inhibited reset to resting Voltage
				if(j==winner and active_pot[j]>layer2[j].Pth):
					#if(np.sum(synapse_memory)==0): ## WINNER NEURON HAS NO PRIOR EXPERIENCE
					x.t_rest = t + x.t_ref
					x.P = Phyperpolarization
					x.Pth-= -1 ## Homoeostasis: Increasing the threshold of the neuron
					count_wins[j]+=1
					for h in range(m):
						for t1 in range(0,t_back-1, -1): # if presynaptic spike came before postsynaptic spike
							if 0<=t+t1<T+1:
								if train[h][t+t1] == 1:
									synapse[j][h] = update(synapse[j][h], rl(t1))
									synapse_memory[j][h]=1
									continue
						if synapse_memory[j][h]!=1:
									synapse[j][h] = update(synapse[j][h], rl(2))
					for p in range(n):
						if p!=winner:
							layer2[p].inhibit()
						#print("Weights updated : ",np.sum(synapse_memory))
					# else:
					# 	for h in range(m):
					# 		for t1 in range(0,t_back-1, -1): # if presynaptic spike came before postsynaptic spike
					# 			if 0<=t+t1<T+1:
					# 				if train[h][t+t1] == 1:
					# 					if(synapse[j][h-1]==1 or synapse[j][h]==1 or  synapse[j][h+1]==1):
					# 						tmp_synapse[j][h] = update(synapse[j][h], rl(t1))
					# 						tmp_synapse_memory[j][h]=1
					# 		if tmp_synapse_memory[j][h]!=1:
					# 					tmp_synapse[j][h] = update(synapse[j][h], rl(2))
					# 	#print("Expected Change = ",abs(np.sum(tmp_synapse)-np.sum(synapse)))
					# 	if(abs(np.sum(tmp_synapse)-np.sum(synapse))<= synaptic_change_threshold):
					# 		x.t_rest = t + x.t_ref
					# 		x.P = Phyperpolarization
					# 		x.Pth-= -1 ## Homoeostasis: Increasing the threshold of the neuron
					# 		count_wins[j]+=1
					# 		synapse=tmp_synapse
					# 		for p in range(n):
					# 			if p!=winner:
					# 				layer2[p].inhibit()
					# 	else: ## NO UPDATION IN WEIGHTS, NEURON SELF-INHIBITS
					# 		print(str(j)+ " Self inhibits")
					# 		x.inhibit()

				elif(winner==False):
					# for p in range(n):
					# 	for h in range(m):
					# 		synapse[p][h] = update(synapse[p][h], rl(-5))
					continue



			winner=False
		
		
		for p in range(n):
				layer2[p].initial()
		
		print(i+" WInner COunt = ",count_wins)
		print("LEARNING NEURON ",np.argmax(count_wins))
		
	for p in range(n):
			reconst_weights(synapse[p],str(p)+"_"+str(k)+"_"+str(i))
			time.sleep(1)
#synapse=np.where(synapse >= np.max(synapse)-2*np.std(synapse), 1, 0)
np.savetxt("weights.csv", synapse, delimiter=",")
ttt = np.arange(0,len(pot_arrays[0]),1)
P_th = []
for i in range(len(ttt)):
	P_th.append(layer2[0].Pth)

# #plotting 
# for i in range(n):
# 	axes = plt.gca()
# 	axes.set_ylim([-100,0])
# 	plt.plot(ttt,P_th, 'r' )
# 	plt.plot(ttt,pot_arrays[i])
# 	plt.show()

# #Reconstructing weights to analyse training
# for i in range(n):
# 	reconst_weights(synapse[i],i)