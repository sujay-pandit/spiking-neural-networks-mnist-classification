########################################################## README ###########################################################

# This file implements STDP curve and weight update rule

##############################################################################################################################



import numpy as np
from matplotlib import pyplot as plt
from parameters import *

#STDP reinforcement learning curve
def rl(t):
	
	if t>0:
		return -A_plus*[np.exp(-float(t)/tau_plus)-STDP_offset]
	if t<=0:
		return A_minus*[np.exp(float(t)/tau_minus)-STDP_offset]


#STDP weight update rule
def update(w, del_w):
	if del_w<0:
		return w + sigma*del_w*(w-abs(w_min))**mu
	elif del_w>0:
		return w + sigma*del_w*(w_max-w)**mu


	