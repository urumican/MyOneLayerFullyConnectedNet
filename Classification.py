import numpy
import MyOneLayerFullyConnectedNet
import cPickle
from scipy.special import expit

def identityFunc(x):
	return x;
# end #

def identityFuncPrime(x):
	return 1
# end # 

def relu(x):
	if x > 0:
		return x
	else:
		return 0.0
# end #

def relu_prime(x):
	if x > 0:
		return 1.0
	elif x == 0:
		return 0.5
	else: 
		return 0.0
# end #


def crossEntropy(x, y):
	return y * numpy.log(x) + (1.0 - y) * log(1.0 - x)
# end #

def corssEngtropyPrime(x, y):
	return (y / x) - (1.0 - y) / (1.0 - x)
# end # 

def sigmoid(x):
	return (1.0 / (1.0 + numpy.exp(-x)))
# end #

def sigmod_prime(x):
	return numpy.exp(-x) / (1.0 + numpy.exp(-x))**2



def main():
	# Import data
	dic = cPickle.load(open("cifar_2class_py2.p","rb"))
	train_data = dic['train_data']
	train_label = dic['train_label']
	test_data = dic['test_data']
	test_label = dic['test_label']

	# Create layer-level parameter
	# Parameter format:
	# [num, activation function, derivative of activation function]
	specification = ((train_data.shape[2], 0, 0), (10, relu, relu_prime), (1, expit, sigmoid_prime))	
	
	# Create new net
	net = MyOneLayerfullyConnectedNet(train_data, train_label, specification, crossEntropy, crossEntropyPrime)
	net.stochasticMiniBatchGradientDescentWithMomentum()
	
# end #
