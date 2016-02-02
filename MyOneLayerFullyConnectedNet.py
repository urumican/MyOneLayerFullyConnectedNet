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
class MyOneLayerFullyConnectedNet(Object):

	## Param's format:
	## 	Param is a tupple. The number of elements in this tuple
	##      equals to the number of layers. Each element is a tuple
	##	containing three kind of information, number of neurons,
	##      activation function, and derivative of activation function.	
	def __init__(self, train_data, train_label, param, loss, lossPrime):
		self.train_data = train_data # input data are all row vector
		self.train_label = train_label # output
		self.numOfLayers = len(param)
		self.numOfNeuronsForAllLayers = [attr[0] for attr in param] # one-hidden 
		self.activationFunction = [attr[1] for attr in param]
		self.activationPrime = [attr[2] for attr in param]
		self.lossFunction = loss
		self.lossPrime = lossPrime
		self.build();
	## end ##

	def build(self):
		self.weights = [] # Weights matrices for all layers.
		self.biases = [] # All biases of all layers.
		self.imputs = []
		self.outputs = []
		self.errors = []
		
		# Initiate Weights using variable 
		for layer in range(self.numOfHiddenLayers - 1):
			# Weight matrices should be m-by-n 
			m = self.numOfNeuronsForAllLayers[layer] # input dim
			n =  self.numOfNeuronsForAllLayers[layer+1] # output dim
			self.weights.append(numpy.random.normal(0, 1, (m, n)))
			self.biases.append(numpy.random.normal(0, 1, (1, n)))
			self.inputs.append(numpy.zeros((1, m)))
			self.outputs.append(numpy.zeros((1, m)))
			self.errors.append(numpy.zeors((1, m)))
		# end #

		# last layer is output 
		n = self.numOfNeuronsForAllLayers[-1]
		self.imputs.append(numpy.zeros((1, n)))
		self.outputs.append(numpy.zeros((1, n)))
		self.errors.append(numpy.zeros((1, n)))
	## end ##

	def feedForward(self, x):
		dim = len(x) 
		x.shape = (dim, 1)
		# Populate input
		self.imputs[0] = x
		self.outputs[0] = x
		for i in range(1, self.numOfLayers):
			self.inputs[i] = numpy.dot(self.inputs[i-1], self.weights[i-1]) + self.bias[i-1] 
			self.outputs[i] = [self.activationFunction[i](n) for n in self.inputs[i]]
		# end #
		return self.outputs[-1] # output the final result
	## end ##

	def updateWeights(self, dataBatch, labelBatch, batchSize, stepSize, gamma):
		# Define a empty increment matrix for momentum
		delta_w = [numpy.zeros(w.shape) for w in self.weights]
		delta_b = [numpy.zeros(b.shape) for b in self.biases]
		# Generate Increments
		for i in range(batchSize):
			data = databatch[i]
			label = labelBatch[i]
			# Go forward
			self.feedForward(data)
			# Go backward
			nabla_weights, nabla_biases = self.backPropagate(label)
			# Accumulate Gradients for this batch
			for layer in xrange(self.numOfLayers - 2) 
				delta_w_t[layer] += nabla_w[layer]
				delta_b_t[layer] += nabla_b[layer]
			# end #
		# end #
		
		return (delta_w, delta_b)

		#self.weights = [w - dw * (stepSize / batchSize) for w, dw in zip(self.weights, delta_w)]
		#self.biases = [b - db * (stepSize / batchSize) for b, db in zip(self.biases, delta_b)]
		
	## end ##

	def stochasticMiniBatchGradientDescentWithMomentum(self, miniBatchSize = 100, stepSize = 1, epoch = 1000, gamma = 0.5):
		# Get the size of the data. 
		dataSize = self.train_data.shape[0]
		for itr in range(epoch):
			randSerie = numpy.random.randint(dataSize, size = dataSize)
			numOfBatch = dataSize / miniBatchSize
			# Create Momentum variable
			momentum_w = [numpy.zeros(w.shape) for w in this.weights]
			momentum_b = [numpy.zeros(b.shape) for b in this.biases] 
			# Start batch gradient descent
			for i in range(numOfBatch)
				# Extract my mini-batch randomly
				miniBatchData = self.train_data[randSerie[i * miniBatchSize : i * miniBatchSize + miniBatchSize - 1]]
				miniBatchLabel = self.train_label[randSerie[i * miniBatchSize : i * miniBatchSize + miniBatchSize - 1]]
				# Get Increment
				delta_w, delta_b = self.updateWeights(miniBatchsize, miniBatchData, miniBatchLabel, stepSize)
				momentum_w = [gamma * mw - stepSize * dw for dw, mw in delta_w, momentum_w]
				momentum_b = [gamma * mb - stepSize * db for db, mb in delta_b, momentum_b]	
				# Updata weights
				self.weights = [w + mw for w, mw in self.weights, momentum_w]
				self.biases = [b + mb for b, mb in self.biases, momentum_b]
			# end #
		# end #
	## end ##

	## My backpropagation is used to calculate all errors at once. ##
	def backPropagate(self, y):
		#initialize matrices for derivative of W and b
		nabla_weights = [numpy.zeros(w.shape) for w in self.weights]
		nabla_biases = [numpy.zeros(b.shape) for b in self.biases]
		# Calculate error for the output layer
		self.errors[-1] = [self.lossPrime(o, l) * self.activationPrime[-1](i) for i, o, l, in zip(self.inputs[-1], self.outputs[-1], y)] 
		nabla_bias[-1] = self.errors[-1]
		nabla_weights[-1] = numpy.outer(self.outputs[-2], self.errors[-1])
		
		# Start backPropagation, calculate from the second-last layers.
		for layer in xrange(self.numOfLayer - 2, 0, -1):
			# Note that the w_l equals to the matrix corresponding to the index 'l-1.
			activPrime = numpy.array([self.activationPrime(i) for i in self.inputs[layer]])
			self.errors[layer] = numpy.dot(self.weights[layer], self.errors[layer+1]) * activPrime
			# Calcualte nabla_Loss / nabla_w_l. 
			nabla_weights[layer] = numpy.outer(self.output[layer-1], self.errors[layer])
			# Calculate nabla_loss / nabla_b_l.
			nabla_biases[layer] = self.errors[layer]
		# end #	
		
		return (nabla_weights, nabla_biases)

	## end ##
	
	## Data should be predicted one by one.
	## I do not support batch prediction for now	
	def prediction(self, data):
		out = self.feedForward(data)
		if out > 1 - out:
			return 1
		else
			return 0
	## end ##

### end ###
