######################################################## README #############################################################

# This file generates rate based spike train from the potential map.

############################################################################################################################


import numpy as np
from numpy import interp
from neuron import neuron
import random
from matplotlib import pyplot as plt
from recep_field import rf
import imageio
from rl import rl
from rl import update
import math
from parameters import *

def encode(pot):

	#initializing spike train
	train = []

	for l in range(pixel_x):
		for m in range(pixel_x):
		
			temp = np.zeros([(T+1),])

			#calculating firing rate proportional to the membrane potential
			freq = interp(pot[l][m], [np.min(pot),np.max(pot)], [1,87])
			
			# print freq
			# if freq<=0:
			# 	print error
				
			freq1 = math.ceil(T/freq)

			#generating spikes according to the firing rate
			k = freq1
			if(pot[l][m]>0):
				while k<(T+1):
					temp[int(k)] = 1
					k = k + freq1
			train.append(temp)
			# print sum(temp)
	return train

# if __name__  == '__main__':
# 	# m = []
# 	# n = []
# 	img = imageio.imread("mnist1/6/" + str(15) + ".png", 0)

# 	pot = rf(img)

# 	# for i in pot:
# 	# 	m.append(max(i))
# 	# 	n.append(min(i))

# 	# print max(m), min(n)
# 	train = encode(pot)
# 	f = open('look_ups/train6.txt', 'w')
# 	print np.shape(train)

# 	for i in range(201):
# 		for j in range(784):
# 			f.write(str(int(train[j][i])))
# 		f.write('\n')

# 	f.close()