import numpy

##########  My Activation Functions ##################
class ActivationFuctions(Object):
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

# end

##########  My Loss Functions ########################
class LossFunctions(Object): # Functions inside should work on scaler
	def crossEntropy(self, z, y):
		return y * numpy.log(z) + (1.0 - y) * log(1.0 - z)
	# end

# end

########### My Neural Network ########################
class MyOneLayerFullyConnectedNet(Object):
	def __init__(self, x, y, param):
		self.x = x # input
		self.y = y # output
		self.numOfLayers = len(param)
		self.numOfNeuronsForAllLayers = [attr[0] for attr in param] # one-hidden 
		self.activitionFunction = [attr[1] for attr in param]
		self.prime = [attr[2] for attr in param]
		self.build();
	# end

	def build(self):
		self.weights = [] # Weights matrices for all layers.
		self.biases = [] # All biases of all layers.
		self.imputs = []
		self.outputs = []
		self.errors = []
		# Initiate Weights using variable 
		for layer in range(self.numOfHiddenLayers + 1):
			# Weight matrices should be m-by-n 
			n = self.numOfNeuronsForAllLayers[layer]
			m =  self.numOfNeuronsForAllLayers[layer+1]
			self.weights.append(numpy.random.normal(0, 1, (m, n)))
			self.biases.append(numpy.random.normal(0, 1, (m, 1)))
			self.inputs.append(numpy.zeros((n, 1)))
			self.outputs.append(numpy.zeros((n, 1)))
			self.errors.append(numpy.zeors((n, 1)))
		# last layer is output 
		n = self.size[-1]
		self.imputs.append(numpy.zeros((n, 1)))
		self.outputs.append(numpy.zeros((n, 1)))
		self.errors.append(numpy.zeros((n, 1)))
	# end

	def feedForward(self, x):
		dim = len(x) 
		x.shape = (dim, 1)
		# Populate input
		self.imputs[0] = x
		self.outputs[0] = x
		for i in range(1, self.numOfLayers):
			self.inputs[i] = slef.weights[i-1].dot(self.inputs[i-1]) + self.biases[i-1]
			self.outputs[i] = self.activationFunction[i](self.inputs[i])
		return self.outputs[-1] # output the final result
	# end

	def weightUpDate(self, x, y): # this is where backprobagation lies.
		output = self.feedForward(x)
		self.errors[-1] = self.prime[-1](self.output[-1]) * (output - y) 
	# end

	def backPropagate(): 

# end

########### My other functions  ######################


