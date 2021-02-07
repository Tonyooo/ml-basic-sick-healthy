## Source: https://www.youtube.com/watch?v=fFUN_x0e_uI

import numpy as np
import matplotlib.pyplot as plt

def init_variables():
	"""
		Init model vars (weights, bias)
	"""
	weights= np.random.normal(size=2)
	bias = 0
	return weights, bias


def get_dataset():
	"""
		Generate dataset
			2 features per instance: white globule & red globule
			For each instance, a target is set (0 = healthy, 1 = sick)
	"""
	# Number of entry per class (X sicks & X healthy)
	row_per_class = 100
	sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
	healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])
	
	features = np.vstack([sick, healthy])
	targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))
	
	return features, targets
	
def pre_activation(features, weights, bias):
	"""
		Compute pre activation
	"""
	return np.dot(features, weights) + bias
	
def activation(pre_activation_result):
	"""
		Activation (sigmooïd)
	"""
	return 1 / (1 + np.exp(-pre_activation_result))
	
def derivate_activation(z):
	# Sigmoid derivative
	return activation(z) * (1 - activation(z))

def predict(features, weights, bias):
	z = pre_activation(features, weights, bias)
	a = activation(z)
	return np.round(a)
	
def cost(predictions, targets):
	return np.mean((predictions - targets)**2)

def train(features, targets, weights, bias):
	"""
		Train model
	"""
	epochs = 100
	learning_rate = 0.1
	
	# Print accuracy
	predictions = predict(features, weights, bias)
	print ("Accuracy:", np.mean(predictions == targets))
	
	# Plot point
	#plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
	#plt.show()
	
	for epoch in range(epochs):
		if epoch % 10 == 0:
			predictions = activation(pre_activation(features, weights, bias))
			print("Cost:", cost(predictions, targets))
		# Init gradients
		weights_gradients = np.zeros(weights.shape) # 2 weights, so create a vector of 2 "0"
		bias_gradient = 0
		# Go through each row
		for feature, target in zip(features, targets):
			# Compute predictions
			z = pre_activation(feature, weights, bias)
			a = activation(z)
			#print ("Prediction for feature '", feature, " => ", a)
			# Update gradient
			weights_gradients += (a - target) * derivate_activation(z) * feature
			bias_gradient += (a - target) * derivate_activation(z)
			#print("Features = ", feature, " - weights = ", weights, " - gradients weights = ", weights_gradients)
		# Update variables
		weights = weights - learning_rate * weights_gradients
		bias = bias - learning_rate * bias_gradient
		
	predictions = predict(features, weights, bias)
	print ("Accuracy:", np.mean(predictions == targets))

if __name__ == '__main__':
	# Generate dataset
	features, targets = get_dataset()
	# Generate vars (weights & bias)
	weights, bias = init_variables()
	# Compute pre-activation
	# z = pre_activation(features, weights, bias)
	# Compute activation
	# a = activation(z)
	train(features, targets, weights, bias)
	# print ("features: \n", features)
	# print ("targets: \n", targets)
	# print ("weights: \n", weights)
	# print ("bias: ", bias)
	# print ("preactivation (feature * weights + bias): \n", z)
	# print ("activation (sigmoid(preactivation)): \n", a)
	pass