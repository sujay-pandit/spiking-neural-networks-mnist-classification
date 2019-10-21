############################################################ README ##############################################################

# This is neuron class which defines the dynamics of a neuron. All the parameters are initialised and methods are included to check
# for spikes and apply lateral inhibition.

###################################################################################################################################

import numpy as np
import random
from matplotlib import pyplot as plt
from parameters import *

class neuron:
	# issue: What if multiple neurons went above threshold but only one will have the highest value?
	# def check(self):
	# 	if self.P>= self.Pth:
	# 		self.P = Prest
	# 		return 1	
	# 	elif self.P < Pinhibit:
	# 		self.P  = Prest
	# 		return 0
	# 	else:
	# 		return 0
	def inhibit(self):
		self.P  = Pinhibit
	def initial(self):
		self.Pth = Pth
		self.t_rest = -1
		self.t_ref=5 #(us)
		self.P = Prest