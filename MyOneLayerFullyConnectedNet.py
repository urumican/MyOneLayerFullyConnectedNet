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
		self.build();
	## end ##

	def build(self):
		self.weights = [] # Weights matrices for all layers.
		self.biases = [] # All biases of all layers.
		self.inputs = []
		self.outputs = []
		self.errors = []
		self.nabla_weights = []
		self.nabla_biases = []
		self.dw = []
		self.db = []
		
		# Initiate Weights using variable 
		for layer in range(self.numOfLayers - 1):
			# Weight matrices should be m-by-n 
			m = self.numOfNeuronsForAllLayers[layer] # input dim
			n =  self.numOfNeuronsForAllLayers[layer+1] # output dim
			self.weights.append(numpy.random.normal(0, 1, (m, n)))
			self.biases.append(numpy.random.normal(0, 1, (1, n)))
			self.nabla_weights = [numpy.zeros(w.shape) for w in self.weights]
			self.nabla_biases = [numpy.zeros(b.shape) for b in self.biases]
			self.inputs.append(numpy.zeros((1, m)))
			self.outputs.append(numpy.zeros((1, m)))
			self.errors.append(numpy.zeros((1, m)))
		# end #

		# last layer is output 
		n = self.numOfNeuronsForAllLayers[-1]
		self.inputs.append(numpy.zeros((1, n)))
		self.outputs.append(numpy.zeros((1, n)))
		self.errors.append(numpy.zeros((1, n)))
	## end ##

	def feedForward(self, x): 
		for layer in range(self.numOfLayers ):
			self.inputs[layer].fill(0)
			self.outputs[layer].fill(0)
			self.errors[layer].fill(0)
		# end #

		# Populate input
		self.inputs[0] = x
		self.outputs[0] = x
		for i in range(1, self.numOfLayers):
			self.inputs[i] = numpy.dot(self.inputs[i-1], self.weights[i-1]) + self.biases[i-1]
			self.outputs[i] = self.activationFunction[i](self.inputs[i])
		# end #
		return self.outputs[-1] # output the final result
	## end ##

	## My backpropagation is used to calculate all errors at once. ##
	def backPropagate(self, y):
		#initialize matrices for derivative of W and b
		
		# Calculate error for the output layer
		self.errors[-1] = self.outputs[-1] - y
	
		nabla_biases[-1] = self.errors[-1]
		nabla_weights[-1] = self.outputs[-2].transpose() * (self.outputs[-1] - y)
		for layer in range(2, self.numOfLayers):
			activPrime  = self.activationPrime[-layer](self.inputs[-layer])
			self.errors[-layer] = numpy.multiply(numpy.dot(self.errors[-(layer - 1)], self.weights[-(layer - 1)].transpose()), activPrime)
			#print 'errors', -layer, self.errors[-layer].shape
			nabla_weights[-layer] = numpy.outer(self.outputs[-(layer + 1)], self.errors[-layer])
			#print 'nabla_weight', -layer, nabla_weights[-layer].shape
			nabla_biases[-layer] = self.errors[-layer]

		return (nabla_weights, nabla_biases)
	## end ##

	def updateWeights(self, batchSize, dataBatch, labelBatch, stepSize):
		# Define a empty increment matrix for momentum
		delta_w = [numpy.zeros(w.shape) for w in self.weights]
		delta_b = [numpy.zeros(b.shape) for b in self.biases]
		# Generate Increments
		for i in range(batchSize):
			data = numpy.array(dataBatch[i])
			#data.shape = (1,data.shape[0])
			label = labelBatch[i]
			# Go forward
			self.feedForward(data)

			#print 'Second layer activation: ', self.outputs[-2]
			#print 'Second layer inputs: ', self.inputs[-2]		
	
			
			# Go backward
			nabla_weights, nabla_biases = self.backPropagate(label)
			print 'nabla_weights: ', nabla_weights
			#print 'Second Layer Gradient: \n', len(nabla_weights) 
			# Accumulate Gradients for this batch
			for layer in range(self.numOfLayers - 1):
				#print 'delta_w', layer, delta_w[layer].shape
				#print 'nabla_weights', layer, nabla_weights[layer].shape
				delta_w[layer] += nabla_weights[layer] / batchSize
				delta_b[layer] += nabla_biases[layer] / batchSize
			# end #
		# end #
		
		return (delta_w, delta_b)

		#self.weights = [w - dw * (stepSize / batchSize) for w, dw in zip(self.weights, delta_w)]
		#self.biases = [b - db * (stepSize / batchSize) for b, db in zip(self.biases, delta_b)]
		
	## end ##

	def stochasticMiniBatchGradientDescentWithMomentum(self, miniBatchSize = 100, stepSize = 1, epoch = 1000, gamma = 0.7):
		# Get the size of the data. 
		dataSize = self.train_data.shape[0]
		for itr in range(epoch):
			print 'Epoch:', itr
			randSerie = numpy.random.randint(dataSize, size = dataSize)
			numOfBatch = dataSize / miniBatchSize
			# Create Momentum variable
			momentum_w = [numpy.zeros(w.shape) for w in self.weights]
			momentum_b = [numpy.zeros(b.shape) for b in self.biases] 
			
			# Start batch gradient descent
			for i in range(numOfBatch):
				# Extract my mini-batch randomly
				miniBatchData = self.train_data[randSerie[i * miniBatchSize : i * miniBatchSize + miniBatchSize]]
				miniBatchLabel = self.train_label[randSerie[i * miniBatchSize : i * miniBatchSize + miniBatchSize]]
				#print 'batch size', miniBatchData.shape
				miniBatchData.shape = (miniBatchSize,3072)
				
				# Get Increment
				delta_w, delta_b = self.updateWeights(miniBatchSize , miniBatchData, miniBatchLabel, stepSize)
				#print 'Gradients: ', delta_w
				for idx in range(len(delta_w)):
					momentum_w[idx] = gamma * momentum_w[idx] - stepSize * delta_w[idx]
					momentum_b[idx] = gamma * momentum_b[idx] - stepSize * delta_b[idx]
				# end #	
				
				#print 'momentum_w', momentum_w		
				#print 'delta_w', delta_w
		
				# Updata weights
				for k in range(len(momentum_w)):
					self.weights[k] = (self.weights[k] + momentum_w[k])
					self.biases[k] = self.biases[k] + momentum_b[k]
				# end #

			# end #

			for k in range(len(momentum_w)):
				self.weights[k] = self.weights[k] / numOfBatch
				self.biases[k] = self.biases[k] / numOfBatch
			# end #



			counter = 0
			batchLoss = 0
			for j in range(10000):
				c, out = self.prediction(self.train_data[j])
				#print 'output: ', out
				batchLoss = batchLoss + self.lossFunction(out, self.train_label[j])
				if c == self.train_label[j]:
					counter = counter + 1
				# end #
			# end # 
			batchLoss = batchLoss / 10000.0

			# Test data part#
			testCount = 0
			testLoss = 0
			for a in range(2000):
				c, out = self.prediction(self.test_data[a])
				testLoss = testLoss + self.lossFunction(out, self.test_label[a])
				if c == self.test_label[a]:
					testCount = testCount + 1

			print 'Output:', self.outputs[-1]
			print 'Second layer activation: ', self.outputs[-2]
			print 'Second layer inputs: ', self.inputs[-2]
			print 'Weights: \n', self.weights
			print 'Loss: ', batchLoss
			print 'testLoss', testLoss / 2000.0
			print 'Acc:', counter / 10000.0
			print 'testAcc: ', testCount / 2000.0	
		# end #
	## end ##
	
	## Data should be predicted one by one.
	## I do not support batch prediction for now	
	def prediction(self, data):
		out = self.feedForward(data)
		loss = self.lossFunction
		if out > 1 - out:
			return (1, out)
		else:
			return (0, out)
	## end ##

### end ###
