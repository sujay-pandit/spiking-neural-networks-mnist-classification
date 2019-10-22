############################################################ README ##############################################################

# This is neuron class which defines the dynamics of a neuron. All the parameters are initialised and methods are included to check
# for spikes and apply lateral inhibition.

###################################################################################################################################

from parameters import *

class neuron:
	def hyperpolarization(self,t):
		self.P = Phyperpolarization
		self.t_ref+=t
	def inhibit(self):
		self.P  = Pinhibit
	def initial(self):
		self.Pth = Pth
		self.t_rest = -1
		self.t_ref=10 #(us)
		self.P = Prest