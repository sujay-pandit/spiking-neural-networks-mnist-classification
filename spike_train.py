######################################################## README #############################################################

# This file generates rate based spike train from the potential map.

############################################################################################################################


import numpy as np
from numpy import interp
import math
from parameters import *

def encode(pot):

	#initializing spike train
	train = []

	for l in range(pixel_x):
		for m in range(pixel_x):
		
			temp = np.zeros([(T+1),])

			#calculating firing rate proportional to the membrane potential
			freq = interp(pot[l][m], [np.min(pot),np.max(pot)], [1,64])

				
			time_period = math.ceil(T/freq)

			#generating spikes according to the firing rate
			time_of_spike = time_period
			if(pot[l][m]>0):
				while time_of_spike<(T+1):
					temp[int(time_of_spike)] = 1
					time_of_spike+= time_period
			train.append(temp)
	return train

