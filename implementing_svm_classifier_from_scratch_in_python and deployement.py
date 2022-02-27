# -*- coding: utf-8 -*-
""" Implementing SVM Classifier from Scratch in Python.ipynb 

Created on Tue Feb 18 09:34:12 2022
@author: kal ab
"""


**SVM Classifier**

Equation of the Hyperplane:

**y = wx - b**

**Gradient Descent:**

Gradient Descent is an optimization algorithm used for minimizing the loss function in various machine learning algorithms. It is used for updating the parameters of the learning model.

w  =  w - α*dw

b  =  b - α*db

**Learning Rate:**

Learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function.

Importing the Dependencies
"""

# importing numpy library
!pip install numpy == 1.21.5
import numpy as np

np. __version__

"""Support Vector Machine Classifier"""

class SVM_classifier():


  # initiating the hyperparameters
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter


  
  # fitting the dataset to SVM Classifier
  def fit(self, X, Y):

    # m  --> number of Data points --> number of rows
    # n  --> number of input features --> number of columns
    self.m, self.n = X.shape

    # initiating the weight value and bias value

    self.w = np.zeros(self.n)

    self.b = 0

    self.X = X

    self.Y = Y

    # implementing Gradient Descent algorithm for Optimization

    for i in range(self.no_of_iterations):
      self.update_weights()



  # function for updating the weight and bias value
  def update_weights(self):

    # label encoding
    y_label = np.where(self.Y <= 0, -1, 1)



    # gradients ( dw, db)
    for index, x_i in enumerate(self.X):

      condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

      if (condition == True):

        dw = 2 * self.lambda_parameter * self.w
        db = 0

      else:

        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
        db = y_label[index]


      self.w = self.w - self.learning_rate * dw

      self.b = self.b - self.learning_rate * db



  # predict the label for a given input value
  def predict(self, X):

    output = np.dot(X, self.w) - self.b
    
    predicted_labels = np.sign(output)

    y_hat = np.where(predicted_labels <= -1, 0, 1)

    return y_hat

"""Importing the Dependencies"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""Data Collection & Processing"""

# loading the data from csv file to pandas dataframe
diabetes_data = pd.read_csv('/content/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# print the first 5 rows of the dataframe
diabetes_data.head()

# number of rows and columns in the dataset
diabetes_data.shape

# getting the statistical measures of the dataset
diabetes_data.describe()

diabetes_data.isnull().sum()



diabetes_data['Diabetes_binary'].value_counts()

"""0 --> Non-diabetic

1 --> Diabetic
"""

# separating the features and target

features = diabetes_data.drop(columns='Diabetes_binary', axis=1)

target = diabetes_data['Diabetes_binary']

print(features)

print(target)

"""Data Standardization"""

scaler = StandardScaler()

scaler.fit(features)

standardized_data = scaler.transform(features)

print(standardized_data)

features = standardized_data
target = diabetes_data['Diabetes_binary']

print(features)
print(target)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state = 2)

print(features.shape, X_train.shape, X_test.shape)

"""Training the Model

Support Vector Machine Classifier
"""

classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=2000, lambda_parameter=0.01)

# training the SVM classifier with training data
classifier.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score on training data = ', training_data_accuracy)

# accuracy on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on test data = ', test_data_accuracy)

"""Building a Predictive System"""

input_data = (1.0,1.0,1.0,35.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,5.0,15.0,15.0,0.0,1.0,11.0,2.0,1.0)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')

else:
  print('The Person is diabetic')
  
#for deployement purpose: loading the svm_classifier into the pickle

import pickle

filename = 'trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))

loaded_model=pickle.load(open('trained_model.sav','rb'))



input_data = (1.0,1.0,1.0,33.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,4.0,0.0,30.0,1.0,1.0,11.0,6.0,7.0)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')

else:
  print('The Person is diabetic')

