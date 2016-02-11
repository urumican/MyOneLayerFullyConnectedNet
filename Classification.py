import numpy
from MyOneLayerFullyConnectedNet import *
import cPickle
from scipy.special import expit

def relu(x):
	tmp = x
	#print tmp
	tmp[numpy.where(tmp <= 0)] = 0;
	return tmp
# end #

def relu_prime(x):
	tmp = x;
	tmp[numpy.where(tmp <= 0)] = 0
	tmp[numpy.where(tmp > 0)] = 1
	return tmp
# end #


def crossEntropy(x, y):
	#print 'output:', x
	return - y * numpy.log(x) - (1.0 - y) * numpy.log(1.0 - x)
# end #

def corssEngtropyPrime(x, y):
	return (y - x) / (x * (1.0 - x))
# end # 

def sigmoid(x):
	return (1.0 / (1.0 + numpy.exp(-x)))
# end #

def sigmoid_prime(x):
	return numpy.exp(-x) / (1.0 + numpy.exp(-x))**2


def main():
	# Import data
	dic = cPickle.load(open("cifar_2class_py2.p","rb"))
	train_data = (dic['train_data'] - dic['train_data'].mean(0)) / dic['train_data'].std(0) 
	train_data.shape = (10000, 3072)
	#train_data = dic['train_data'] / 255
	train_label = dic['train_labels']	
	test_data = (dic['test_data'] - dic['test_data'].mean(0)) / dic['test_data'].std(0)
	test_data.shape = (2000, 3072)
	#test_data = dic['test_data'] / 255
	test_label = dic['test_labels'] 

	# Create layer-level parameter
	# Parameter format:
	# [num, activation function, derivative of activation function]
	specification = ((train_data.shape[1], 0, 0), (10, relu, relu_prime), (1, expit, sigmoid_prime))	
	
	# Create new net
	net = MyOneLayerFullyConnectedNet(train_data, train_label, test_data, test_label, specification, crossEntropy, corssEngtropyPrime)
	net.mySDGwithMomentum()
	
# end #

if __name__ == "__main__":
    main()
