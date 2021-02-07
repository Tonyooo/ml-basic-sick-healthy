## Source: https://www.youtube.com/watch?v=aylST9MEOdw

if __name__ == '__main__':
	# Function to minimize
	fc = lambda x, y: (3*x**2) + (x*y) + (5*y**2)
	
	# Partial derivates
	partial_derivative_x = lambda x, y: (6*x) + y
	partial_derivative_y = lambda x, y: (10*y) + x
	
	# Set variable
	x = 10 # Weights in real life
	y = -13 # Weights in real life
	
	# Learning rate
	learning_rate = 0.1
	print ("Fc =", (fc(x, y)))
	
	# Iterate
	for epoch in range(0, 20):
		# Compute gradients
		x_gradient = partial_derivative_x(x, y)
		y_gradient = partial_derivative_y(x, y)
		# Apply gradient descent
		x = x - (learning_rate * x_gradient)
		y = y - (learning_rate * y_gradient)
		
		print ("Fc =", (fc(x, y)))
		
	# Print final weights
	print ("")
	print ("Final x = ", x)
	print ("Final y = ", y)