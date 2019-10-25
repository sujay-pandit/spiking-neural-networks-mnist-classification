############################################################ README ##############################################################

# This is neuron class which defines the dynamics of a neuron. All the parameters are initialised and methods are included to check
# for spikes and apply lateral inhibition.

###################################################################################################################################

from parameters import *

class neuron:
	def hyperpolarization(self,t):
		self.P = Phyperpolarization
		self.t_rest=t+self.t_ref
	def inhibit(self,t):
		self.P  = Pinhibit
		self.t_rest=t+self.t_ref
	def initial(self):
		self.Pth = Pth
		self.t_rest = -1
		self.t_ref=15 #(us)
		self.P = Prest