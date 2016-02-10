#####################################################
# Li, Xin:                                          #
#####################################################
# The first homework for Deep Learning, which is to #
# compose a one layer fully connected 2-class clas- #
# sification net.                                   #
#                                                   #
# For classification problem, we use sigmoid units  #
# as out put layer.                                 #
#####################################################

import numpy

### My Neural Network ###
class MyOneLayerFullyConnectedNet:

	## Param's format:
	## 	Param is a tupple. The number of elements in this tuple
	##      equals to the number of layers. Each element is a tuple
	##	containing three kind of information, number of neurons,
	##      activation function, and derivative of activation function.
	def __init__(self, train_data, train_label, test_data, test_label, param, loss, lossPrime):
		self.train_data = train_data # input data are all row vector
		self.train_label = train_label # output
		self.test_data = test_data
		self.test_label = test_label
		self.numOfLayers = len(param)
		self.numOfNeuronsForAllLayers = [attr[0] for attr in param] # one-hidden
		self.activationFunction = [attr[1] for attr in param]
		self.activationPrime = [attr[2] for attr in param]
		self.lossFunction = loss
		self.lossPrime = lossPrime
		self.build()
	## end ##

	def build(self):
		# initiate properties.
		self.weights = [] # Weights matrices for all layers.
		self.biases = [] # All biases of all layers.
		self.inputs = []
		self.outputs = []
		self.errors = []
	## end ##

	def feedForward(self, x): # feedForward() is a matrix operation
		## The data will be N-by-3072 matrix, means N data with 3072 dimension 
		self.inputs[0] = x
		self.outputs[0] = x
		for i in range(1, self.numOfLayers):
			self.inputs[i] = numpy.dot(self.outputs[i-1], self.weights[i-1]) + numpy.dot(numpy.ones((x.shape[0], 1)), self.biases[i-1])
			self.outputs[i] = self.activationFunction[i](self.inputs[i])
		# end #
	## end ##

	## My backpropagation is used to calculate all errors at once. ##
	def backPropagate(self, y):
		# Calculate error for the output layer, it is matrix operation
		self.errors[-1] = self.outputs[-1] - y

		self.nabla_biases[-1] = self.errors[-1]
		self.nabla_weights[-1] = numpy.dot(self.outputs[-2].transpose(), self.errors[-1])
		for layer in range(2, self.numOfLayers):
			activPrime  = self.activationPrime[-layer](self.inputs[-layer])
			self.errors[-layer] = numpy.dot(self.errors[-(layer - 1)], self.weights[-(layer - 1)].transpose()) * activPrime # activPrime is k-by-n
			self.nabla_weights[-layer] = numpy.outer(self.outputs[-(layer + 1)].transpose(), self.errors[-layer]) # error at current layer is k-by-n
			self.nabla_biases[-layer] = numpy.dot(numpy.ones((1, y.shape[0])), self.errors[-layer])
	## end ##

	def updateWeights(self, batchSize, dataBatch, labelBatch, stepSize):
		# Go forward
		self.feedForward(dataBatch)

		# Go backward
		self.backPropagate(labelBatch)
	## end ##

	def stochasticMiniBatchGradientDescentWithMomentum(self, miniBatchSize = 100, stepSize = 0.01, epoch = 1000, gamma = 0.7):
		# Get the size of the data.
		dataSize = self.train_data.shape[0]
	
		# setup the network
		self.setup(miniBatchSize)
			
		
		for itr in range(epoch):
			print 'Epoch:', itr
			randSerie = numpy.random.randint(dataSize, size = dataSize)
			numOfBatch = dataSize / miniBatchSize

			# Start batch gradient descent
			for i in range(numOfBatch):
				# Extract my mini-batch randomly
				miniBatchData = self.train_data[randSerie[i * miniBatchSize : i * miniBatchSize + miniBatchSize]]
				miniBatchLabel = self.train_label[randSerie[i * miniBatchSize : i * miniBatchSize + miniBatchSize]]

				# Get Increment
				self.updateWeights(miniBatchSize , miniBatchData, miniBatchLabel, stepSize)

				# Updata weights
				for k in range(len(self.momentum_w)):
					self.weights[k] = (self.weights[k] + self.momentum_w[k])
					self.biases[k] = self.biases[k] + self.momentum_b[k]
				# end #

			# end #

			#for k in range(len(self.momentum_w)):
			#	self.weights[k] = self.weights[k] / numOfBatch
			#	self.biases[k] = self.biases[k] / numOfBatch
			## end #

		# end #
	## end ##

	## Heler method for re-initiate
	def setup(self, batchSize):
		# clear matrices used previously.
		self.weights = [] 
		self.biases = [] 
		self.inputs = []
		self.outputs = []
		self.errors = []

		# Initiate Weights using variable
		for layer in range(self.numOfLayers - 1):
			# Weight matrices should be m-by-n
			m = self.numOfNeuronsForAllLayers[layer] # input dim
			n =  self.numOfNeuronsForAllLayers[layer+1] # output dim
			self.weights.append(numpy.random.normal(0, 1, (m, n)))
			self.biases.append(numpy.random.normal(0, 1, (1, n)))
			self.inputs.append(numpy.zeros((batchSize,n)))
			self.outputs.append(numpy.zeros((batchSize, m)))
			self.errors.append(numpy.zeros((batchSize, m)))
		# end #

		# last layer is output
		n = self.numOfNeuronsForAllLayers[-1]
		self.inputs.append(numpy.zeros((batchSize, n)))
		self.outputs.append(numpy.zeros((batchSize, n)))
		self.errors.append(numpy.zeros((batchSize, n)))
	## end ##

	## Data should be predicted one by one.
	## I do not support batch prediction for now
	def prediction(self, data):
		out = self.feedForward(data)
		if out > 1 - out:
			return (1, out)
		else:
			return (0, out)
	## end ##


### end ###
