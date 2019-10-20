############################################## README #################################################

# This calculates threshold for an image depending upon its spiking activity.

########################################################################################################


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
import os


## Homoeostasis
def threshold(train):

	tu = np.shape(train[0])[0]
	thresh = 0
	for i in range(tu):
		simul_active = sum(train[:,i])
		if simul_active>thresh:
			thresh = simul_active

	return (thresh/3)


# if __name__ == '__main__':	

# 	# img = imageio.imread("mnist1/" + str(1) + ".png", 0)
# 	img = np.array(Image.open("mnist1/" + str(1) + ".png", 0))
# 	print img
# 	# pot = rf(img)
# 	# train = np.array(encode(pot))
# 	# print threshold(train)