#####################################################
#                   Li, Xin                         #                                                   
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
	def __init__(self, train_data, train_label, param):
		self.train_data = train_data # input
		self.train_label = train_label # output
		self.numOfLayers = len(param)
		self.numOfNeuronsForAllLayers = [attr[0] for attr in param] # one-hidden 
		self.activationFunction = [attr[1] for attr in param]
		self.activationPrime = [attr[2] for attr in param]
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
			n = self.numOfNeuronsForAllLayers[layer]
			m =  self.numOfNeuronsForAllLayers[layer+1]
			self.weights.append(numpy.random.normal(0, 1, (m, n)))
			self.biases.append(numpy.random.normal(0, 1, (m, 1)))
			self.inputs.append(numpy.zeros((n, 1)))
			self.outputs.append(numpy.zeros((n, 1)))
			self.errors.append(numpy.zeors((n, 1)))
		# end #

		# last layer is output 
		n = self.size[-1]
		self.imputs.append(numpy.zeros((n, 1)))
		self.outputs.append(numpy.zeros((n, 1)))
		self.errors.append(numpy.zeros((n, 1)))
	## end ##

	def feedForward(self, x):
		dim = len(x) 
		x.shape = (dim, 1)
		# Populate input
		self.imputs[0] = x
		self.outputs[0] = x
		for i in range(1, self.numOfLayers):
			self.inputs[i] = slef.weights[i-1].dot(self.inputs[i-1]) + self.biases[i-1]
			self.outputs[i] = self.activationFunction[i](self.inputs[i])
		# end #
		return self.outputs[-1] # output the final result
	## end ##

	def updateWeights(self, dataBatch, labelBatch, batchSize, stepSize):
		# Define a empty increment matrix
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
			# Start updating
			for layer in xrange(self.numOfLayers - 2) 
				delta_w[layer] += nabla_w[layer]
				delta_b[layer] += nabla_b[layer]
			# end #
		# end #
		self.weights = [w - dw * (stepSize / batchSize) for w, dw in self.weights, delta_w]
		self.biases = [b - db * (stepSize / batchSize) for b, db in self.biases, delta_b]
		
	## end ##

	def stochasticMiniBatchGradientDescent(self, miniBatchSize = 128, stepSize = 1, epoch = 1000):
		dataSize = self.train_data.shape[0]
		randSerie = numpy.random.randint(dataSize, size = dataSize)
		
		for itr in range(epoch)
			# Extract my mini-batch randomly
			miniBatchData = self.train_data[randSerie[itr * miniBatchSize : itr * miniBatchSize + miniBatchSize - 1]]
			miniBatchLabel = self.train_label[randSerie[itr * miniBatchSize : itr * miniBatchSize + miniBatchSize - 1]]
			self.updateWeights(miniBatchsize, miniBatchData, miniBatchLabel, stepSize)
		# end #
	## end ##

	## My backpropagation is used to calculate all errors at once. ##
	def backPropagate(self, y):
		#initialize matrices for derivative of W and b
		nabla_weights = [numpy.zeros(w.shape) for w in self.weights]
		nabla_biases = [numpy.zeros(b.shape) for b in self.biases]
		# Calculate error for the output layer
		self.errors[-1] = self.lossPrime(self.output[-1], y) * self.activationPrime[-1](self.inputs[-1])
		nabla_bias[-1] = self.errors[-1]
		nabla_weights[-1] = numpy.dot(self.errors[-1], self.outputs[-2].T)
		
		# Start backPropagation, calculate from the second-last layers.
		for layer in xrange(self.numOfLayer - 2, 0, -1):
			# Note that the w_l equals to the matrix corresponding to the index 'l-1.
			self.errors[layer] = numpy.dot(self.weights[layer].T, self.errors[layer+1]) * self.activationPrime(self.inputs[layer])
			# Calcualte nabla_Loss / nabla_w_l. 
			nabla_weights[layer] = numpy.dot(self.errors[layer], self.output[layer-1].transpose())
			# Calculate nabla_loss / nabla_b_l.
			nabla_biases[layer] = self.errors[layer]
		# end #	
		
		return (nabla_weights, nabla_biases)

	## end ##
	
	## Data should be predicted one by one.
	## I do not support batch prediction for now	
	def prediction(self, data):
		return self.feedForward(data)

	## end ##

### end ###
