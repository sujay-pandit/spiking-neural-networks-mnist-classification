

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

for k in range(epoch):
	print(k)
	for i in os.listdir("./train_mnist/"):
		img = imageio.imread("./train_mnist/"+i)

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
		f_spike = 0
		winner = False
		count_wins= np.zeros(3)
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
					x.t_rest = t + x.t_ref
					x.P = Phyperpolarization
					x.Pth-= -1 ## Homoeostasis: Increasing the threshold of the neuron
					count_wins[j]+=1
					for h in range(m):
						for t1 in range(0,t_back-1, -1): # if presynaptic spike came before postsynaptic spike
							if 0<=t+t1<T+1:
								if train[h][t+t1] == 1:
									synapse[j][h] = update(synapse[j][h], rl(t1))
									winner_synapses.append(h)
						if h not in winner_synapses:
									synapse[j][h] = update(synapse[j][h], rl(2))
					# for p in range(n):
					# 	if(p!=winner):
					# 		for h in winner_synapses:
					# 			synapse[p][h] = update(synapse[p][h], rl(2))

							
					
				elif(winner==False):
					for p in range(n):
						for h in range(m):
							synapse[p][h] = update(synapse[p][h], rl(-5))
							


					continue
				# else:
				# 	# for h in winner_synapses: ## lower the weights for loser neurons
				# 	layer2[j].Pth-=1
				# 		#synapse[j][h] = update(synapse[j][h], rl(1))
									
					
					
		if(winner!=False):
			for p in range(m):
				if sum(train[p])==0:
					# synapse[winner][p] -= 0.06
					# if(synapse[winner][p]<w_min):
					synapse[winner][p] = w_min
			winner=False
		for p in range(n):
			reconst_weights(synapse[p],str(i)+"_"+str(p))
		for p in range(n):
				layer2[p].initial()
		
		print(i+" WInner COunt = ",count_wins)
		time.sleep(3)
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

#Reconstructing weights to analyse training
for i in range(n):
	reconst_weights(synapse[i],i)