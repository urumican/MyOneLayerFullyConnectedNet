import numpy

##########  My Loss Functions ########################
class LossFunctionsMaster(Object): # Functions inside should work on scaler
	def crossEntropy(self, out, y):
		return y * numpy.log(out) + (1.0 - y) * log(1.0 - out)
	# end

	def crossEntropy_prime(self, out, y):
		return (y / out) - (1.0 - y) / (1.0 - out)
	# end
# end
