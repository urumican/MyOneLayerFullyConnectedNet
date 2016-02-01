import numpy

##########  My Activation Functions ##################
class ActivationFuctionsMaster(Object):
	def relu(self, x):
		if x >= 0:
			return x
		else: 
			return 0.0
	# end	

	def relu_prime(self, x): 
		if x > 0:
			return 1.0
		elif x = 0:
			return 0.5
		else 
			return 0.0
	# end

	def sigmoid(self, x):
		return (1.0 / (1.0 + numpy.exp(-x)))
	# end

	def sigmoid_prime():
		return (1 - self.sigmoid(x)) * self.sigmoid(x)
	# end
# end
