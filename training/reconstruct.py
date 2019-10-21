###################################################### README #####################################################

# This file is used to leverage the generative property of a Spiking Neural Network. reconst_weights function is used
# for that purpose. Looking at the reconstructed images helps to analyse training process.

####################################################################################################################


import numpy as np
from numpy import interp
import imageio
from recep_field import rf
from parameters import *


def reconst_weights(weights, num):
	weights = np.array(weights)
	weights = np.reshape(weights, (pixel_x,pixel_x))
	img = np.zeros((pixel_x,pixel_x))
	for i in range(pixel_x):
		for j in range(pixel_x):
			img[i][j] = int(interp(weights[i][j], [w_min,w_max], [0,255]))	

	imageio.imwrite('neuron_' + str(num) ,img.astype(np.uint8))
	return img

def reconst_rf(weights, num):
	weights = np.array(weights)
	weights = np.reshape(weights, (pixel_x,pixel_x))
	img = np.zeros((pixel_x,pixel_x))
	for i in range(pixel_x):
		for j in range(pixel_x):
			img[i][j] = int(interp(weights[i][j], [-2,3.625], [0,255]))	

	imageio.imwrite('neuron' + str(num) + '.png' ,img)
	return img


# if __name__ == '__main__':

# 	img = imageio.imread("images2/" + "69" + ".png", 0)
# 	pot = rf(img)
# 	reconst_rf(pot, 12)