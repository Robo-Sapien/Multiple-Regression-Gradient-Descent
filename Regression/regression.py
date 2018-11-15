import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(1)

class Regression:
	"""
	class for performing regression
	"""
	def __init__(self,addr,split_ratio,validation_ratio):
		"""
		Class constructor
		
		params:
		@addr: address of data file
		@split_ratio: fraction of data to be used for training the model
		@validation_ration: fraction of data to be used for validation

		attributes:
		@fileaddr: address of the file containing the data
		@data: dataframe containing values of features and value of target var
		@training_size: size of training data
		@validation_size: size of validation data
		@features: value of features for each entry in the data
		@target: value of target variables for each entry in the data

		"""
		self.file_addr = addr
		self.data = pd.read_excel(self.file_addr)
		self.num_features = len(self.data.columns)
		self.data = self.data.values
		self.training_size = split_ratio*self.data.shape[0]
		self.training_size = int(self.training_size)
		self.validation_size = validation_ratio*self.data.shape[0]
		self.validation_size = int(self.validation_size)


		self.features,self.target = self.data[:, :-1], self.data[:, -1]


	def feature_scaling(self):
		"""
		
		function to scale the values of features so that they all acquire values in the same range 
		
		"""
		mean_value = self.features.mean(axis=0)
		std = self.features.std(axis=0)

		print(mean_value)
		print(std)

		self.features = (self.features - mean_value)/std
		x = np.ones((self.data.shape[0],1))
		self.features = np.append(x,self.features,axis=1)

		self.x_train, self.y_train = self.features[:self.training_size,:], self.target[:self.training_size]
		self.x_validate, self.y_validate = self.features[self.training_size:(self.training_size + self.validation_size),:], self.target[self.training_size:(self.training_size + self.validation_size)]
		self.x_test, self.y_test = self.features[(self.training_size + self.validation_size):,:], self.target[(self.training_size + self.validation_size):]

		#print(self.x_train.shape[1])


	def normal_equation(self):
		"""

		Method to use normal equation method for getting value of parameters in regression expression
		
		"""
		#print(self.x_test)

		phi = self.x_train
		phi_transpose = self.x_train.transpose()

		parameters = np.matmul(phi_transpose,phi)
		parameters = np.linalg.inv(parameters)
		parameters = np.matmul(parameters,phi_transpose)
		parameters = np.matmul(parameters,self.y_train)

		#print(parameters)

		output_val = np.matmul(parameters,self.x_test.transpose())

		error = self.error_estimation(output_val,1)

		#print("error == > " + str(error))
		return error


	def gradient_descent_without_regularization(self,iterations,learning_rate):
		"""
		
		Method to estimate the value of parameters in linear regression without
		imposing any restriction on the value of parameters. 
		
		params:
		@iterations: to impose restriction on the number of iterations
		@learning_rate: learning rate to be used in the stochastic gradient descent

		"""
		parameters = np.random.randn(5,)
		num = self.x_train.shape[0]
		#print(num)
		initial_parameters = parameters

		prev_error = 0.0

		while 1>0:
		# for i in range(0,iterations):
			feature = self.x_train
			prediction = np.matmul(initial_parameters,feature.transpose())
			prediction = prediction.transpose()

			#print(prediction)

			for k in range(0,5):
				parameters[k] = parameters[k] - (np.sum((learning_rate/num)*((prediction - self.y_train)*self.x_train[:,k])))
				

			# print(parameters)

			output_val = np.matmul(parameters,self.x_test.transpose())					
			curr_error = self.error_estimation(output_val,1)
			initial_parameters = parameters

			if abs(curr_error - prev_error) < 10**(-3) or abs(curr_error - prev_error) > 10**(6):
				prev_error = curr_error
				break

			prev_error = curr_error

		#print(parameters)		

		# output_val = np.matmul(parameters,self.x_test.transpose())					
		# error = self.error_estimation(output_val)
		
		# print("error => " + str(error))

		return prev_error


	def gradient_descent_with_ridge_regularization(self,iterations,learning_rate,reg_coeff):
		"""

		Method to estimate the value of parameters in linear regression without
		imposing any restriction on the value of parameters.
		
		Ridge regression: imposing restriction on the value of square of parameters

		params:
		@iterations: to impose restriction on the number of iterations
		@learning_rate: learning rate to be used in the stochastic gradient descent
		@reg_coeff: regularization coeff used in ridge regression

		"""
		parameters = np.random.randn(5,)
		num = self.x_train.shape[0]
		initial_parameters = parameters

		prev_error = 0.0

		while 1>0:
		# for i in range(0,iterations):	
			feature = self.x_train
			prediction = np.matmul(initial_parameters,feature.transpose())
			prediction = prediction.transpose()

			for k in range(0,5):
				parameters[k] = parameters[k] - np.sum(((learning_rate/num)*(prediction - self.y_train)*self.x_train[:,k])) - ((learning_rate/num)*(reg_coeff)*parameters[k])
				

			initial_parameters = parameters
			
			reg_error = (learning_rate/num)*reg_coeff*np.sum(np.square(parameters))
			output_val = np.matmul(parameters,self.x_test.transpose())						
			curr_error = self.error_estimation(output_val,1) + reg_error
			
			#print(reg_error)

			if abs(curr_error - prev_error) < 10**(-4) or abs(curr_error - prev_error) > 10**(6):
				prev_error = curr_error
				break

			prev_error = curr_error


		return prev_error

	def gradient_descent_with_lasso_regularization(self,iterations,learning_rate,reg_coeff):
		"""
		
		lasso regression which imposes linear constraints on the values of parameters
		
		Lasso regression: imposing restriction on the absolute value of parameters

		params:
		@iterations: to impose restriction on the number of iterations
		@learning_rate: learning rate to be used in the stochastic gradient descent
		@reg_coeff: regularization coeff used in ridge regression

		"""
		parameters = np.random.randn(5,)
		num = self.x_train.shape[0]
		initial_parameters = parameters

		prev_error = 0.0 

		while 1>0:
			feature = self.x_train
			prediction = np.matmul(initial_parameters,feature.transpose())
			prediction = prediction.transpose()

			for k in range(0,5):
				parameters[k] = parameters[k] - np.sum(((learning_rate/num)*(prediction - self.y_train)*self.x_train[:,k])) - ((learning_rate/num)*(reg_coeff) *(abs(parameters[k])/parameters[k]))
				
			initial_parameters = parameters
			
			reg_error = (learning_rate/num)*reg_coeff*np.sum(abs(parameters))
			output_val = np.matmul(parameters,self.x_test.transpose())
			curr_error = self.error_estimation(output_val,1) + reg_error

			if abs(curr_error - prev_error) < 10**(-4) or abs(curr_error - prev_error) > 10**(6):
				prev_error = curr_error
				break

			prev_error = curr_error


		return prev_error

	def error_estimation(self,output_val,flag):
		"""
		
		estimating the test,train or validation error

		params:
		@output_val: estimated value of target variable
		@flag: variable to indicate whether test, train or validation error is being calculated
				flag=1 => test error
				flag=2 => training error
				flag=0 => validation error 

		"""
		if flag == 1:
			num = self.y_test.shape[0]
		elif flag == 2:
			num = self.y_train.shape[0]
		else:
			num = self.y_validate.shape[0]

		error = 0

		for i in range(0,num):
			#print(str(self.y_test[i]) + " " + str(output_val[i]))
			if flag == 1:
				error = error + (self.y_test[i]-output_val[i])*(self.y_test[i]-output_val[i])
			elif flag == 2:
				error = error + (self.y_train[i]-output_val[i])*(self.y_train[i]-output_val[i])
			else:
				error = error + (self.y_validate[i]-output_val[i])*(self.y_validate[i]-output_val[i])
			#error_1 = error_1 + (self.y_train[i]-output_val[i])*(self.y_train[i]-output_val[i])

		#print(error)

		error = 0.5*(error/num)
		#error = error/num
		#print(num)
		# print(error)

		return error

	def plots_no_regularization(self,X,Y,x_name,y_name):
		"""	
		
		function to generate the plots

		params:
		@X: x axis parameters
		@Y: y axis parameters
		@x_name: name of the x axis
		@y_name: name of the y axis

		"""
		fig = plt.figure()
		fig.suptitle("reg_coeff vs validation_error",fontsize=20)
			
		#print(Y[0])

		for i in range(0,len(X)): 	
			x = X[i]
			y = Y[i]
			#z = Z[i]

			plt.plot(x,y)
			#plt.show()

		plt.plot(X,Y)

		plt.xlabel(x_name)
		plt.ylabel(y_name)
		#plt.legend()

		plt.show()
		#fig.savefig(str(learning_rate)+".png")

	def plots_with_regularization(self,X,Y,reg_coeff,x_name,y_name):
		"""
		
		function to generate the plots

		params:
		@X: x axis parameters
		@Y: y axis parameters
		@reg_coeff: regularization coeff
		@x_name: name of the x axis
		@y_name: name of the y axis

		"""
		fig = plt.figure()
		fig.suptitle("error until convergence",fontsize=14)
		for i in range(0,len(X)):
			x = X[i]
			y = Y[i]
			z = Z[i]

			plt.plot(x,y,label="reg_coeff:" + str(z))


			
		plt.xlabel(x_name)
		plt.ylabel(y_name)
		plt.legend()

		plt.show()
		#fig.savefig("regularization/"+str(learning_rate) + "_" + str(reg_coeff)+".png")


if __name__ == '__main__':

	file_addr = "dataset/Folds5x2_pp.xlsx"
	reg = Regression(file_addr,0.70,0.091)
	reg.feature_scaling()
	error = reg.normal_equation()
	print("normal_equation error =>" + str(error))
	#reg.normal_equation()
	
	# to generate the plot of validation error vs learning rate without regularization

	# X = []
	# Y = []
	# Z = []

	# learning_rate = 0.0001
	# while learning_rate <= 0.01:
	# 	x = []
	# 	y = []
	# 	Z.append(learning_rate)

	# 	iterations = 10
	# 	while iterations <= 1000:
	# 		x.append(iterations)
	# 		error = reg.gradient_descent_without_regularization(iterations,learning_rate)
	# 		iterations = iterations + 10
	# 		y.append(error)

	# 	learning_rate = learning_rate * 10

	# 	X.append(x)
	# 	Y.append(y)

	# reg.plots_no_regularization(X,Y,Z,"iterations","error")


	# to generate the plot of validation error vs reg_coeff for ridge regression and lasso regression 

	# learning_rate = 0.01
	# reg_coeff = 0.0
	# X = []
	# Y = []
	# Z = [] 


	# while reg_coeff <= 50:
	# 	iterations = 10
	# 	#x = []
	# 	#y = []
	# 	#Z.append(reg_coeff)

	# 	# while iterations <= 1000:
	# 	# 	x.append(iterations)
	# 	error = reg.gradient_descent_with_lasso_regularization(iterations,learning_rate,reg_coeff)
	# 	# 	y.append(error)
	# 	# 	iterations = iterations + 10


	# 	if reg_coeff < 1.0:
	# 		reg_coeff = reg_coeff + 0.1
	# 	else:
	# 		reg_coeff = reg_coeff + 5

	# 	X.append(reg_coeff)
	# 	Y.append(error)

	# reg.plots_no_regularization(X,Y,"reg_coeff","valid_error")

	# lasso_error = reg.gradient_descent_without_regularization(10,)


	# learning_rate = 0.001

	# x = []
	# y = []

	# X = []
	# Y = []
	# z = []

	# while learning_rate <= 1:
	# 	error = reg.gradient_descent_without_regularization(0,learning_rate)
	# 	x.append(learning_rate)
	# 	y.append(error)
	# 	#z.append(learning_rate)
	# 	learning_rate = learning_rate + 0.005

	# X.append(x)
	# Y.append(y)

	# reg.plots_no_regularization(X,Y,"learning_rate","validation_error")

	error = reg.gradient_descent_without_regularization(0,0.0815)
	print("No regularization error =>" + str(error))

	error = reg.gradient_descent_with_ridge_regularization(0,0.0815,0.3)
	print("Ridge regression error =>" + str(error))	

	error = reg.gradient_descent_with_lasso_regularization(0,0.0815,0.8)
	print("Lasso regression error =>" + str(error))		

	# while learning_rate<=0.05:
	# 	x = []
	# 	y = []

	# 	iterations = 10

	# 	while iterations<100:
	# 		error = reg.gradient_descent_without_regularization(iterations,learning_rate)
	# 		x.append(iterations)
	# 		y.append(error)

	# 		iterations = iterations + 10

	# 	reg.plots_no_regularization(x,y,learning_rate,"iterations","error")

	# 	learning_rate = learning_rate*5

	#reg.gradient_descent_without_regularization(10000,0.001)

	
	# learning_rate = 0.0001

	# while learning_rate<=0.05:
	# 	reg_coeff = 0.5
	# 	while reg_coeff<=1:
	# 		x = []
	# 		y = []

	# 		iterations = 10

	# 		while iterations < 100:
	# 			error = reg.gradient_descent_with_regularization(iterations,learning_rate,reg_coeff)
	# 			x.append(iterations)
	# 			y.append(error)

	# 			iterations = iterations + 10

	# 		reg.plots_with_regularization(x,y,learning_rate,reg_coeff,"iterations","error")

	# 		reg_coeff = reg_coeff + 0.1

	# 	learning_rate = learning_rate*5

	#reg.gradient_descent_with_ridge_regularization(10000,0.001,1)
	#reg.gradient_descent_with_lasso_regularization(10000,0.001,1)
	