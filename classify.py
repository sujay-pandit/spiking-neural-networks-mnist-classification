##################### README ###################################################
# This file executes the classification algorithm over input testing images.
# Winner neurons inhibit other neurons by a phenomenon called Lateral inhibition
# Spike for each output neuron at each time stamp is monitored.
################################################################################
import numpy as np
from neuron import neuron
import random
from recep_field import rf
import imageio
from spike_train import *
import pandas as pd
import os
from parameters import *
import time

#time series 
time_of_learning  = np.arange(1, T+1, 1)
output_layer = []
# creating the hidden layer of neurons


#synapse matrix
synapse = np.zeros((n,m))
#learned weights
learned_weights=pd.read_csv("weights.csv",header=None)
neuron_labels=pd.read_csv("labels.csv",header=None)
labels_matrix=np.array(neuron_labels.values)
weight_matrix = np.array(learned_weights.values)
m = weight_matrix.shape[1] #Number of neurons in first layer
n = weight_matrix.shape[0] #Number of neurons in second layer
for i in range(n):
	a = neuron()
	output_layer.append(a)
	synapse[i] = weight_matrix[i]


predicted_class=[]
actual_class=[]
prediction_count=np.zeros((n,n))
for folder_number in range(10):
	image_path='./mnist_png/testing/'+str(folder_number)+'/'
	for i in os.listdir(image_path):
		t0 = time.time()
		
		count_spikes = np.zeros((n,1))

		#read the image to be classified
		img = imageio.imread(image_path+i)

		#initialize the potentials of output neurons
		for x in output_layer:
			x.initial()

		#generate spike trains. Select between deterministic and stochastic
		train = np.array(encode(rf(img)))

		#flag for lateral inhibition
		f_spike = 0
		active_pot = np.zeros((n,1))
		winner=False
		for t in time_of_learning:
				for j, x in enumerate(output_layer):
					if(x.t_rest<t):
						x.P = x.P + np.dot(synapse[j], train[:,t])
						if(x.P>Prest):
							x.P -= Pdrop
					active_pot[j] = x.P
				for j in range(n):
					if(j==np.argmax(active_pot)):
						if(active_pot[j]>output_layer[j].Pth):
							count_spikes[j]+=1
							output_layer[j].Pth-=1
							output_layer[j].hyperpolarization(t)
							for p in range(n):
								if p!=j:
									if(output_layer[p].P>output_layer[p].Pth):
										output_layer[p].inhibit(t)
							break
		
		#print(count_spikes)
		print(i)
		print("Predicted_class = ",labels_matrix[np.argmax(count_spikes)][0])
		print("Actual class = ",folder_number)
		print("Time for inference = ",time.time()-t0)
		predicted_class.append(labels_matrix[np.argmax(count_spikes)][0])
		actual_class.append(folder_number)
		prediction_count[int(folder_number)][int(labels_matrix[np.argmax(count_spikes)][0])]+=1
		#print("Prediction for "+str(folder_number)+" : ",prediction_count[folder_number])
accuracy=(np.sum(np.array(predicted_class)==np.array(actual_class))/len(predicted_class))*100
np.savetxt("prediction_matrix.csv",prediction_count,delimiter=',')
print(accuracy)
