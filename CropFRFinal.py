#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:55:01 2023

@author: yashbalkhande
"""

# Dataset :- https://www.kaggle.com/code/nabeelraza/crop-recommendation-project

# importing libraries  
# We are importing pandas Library as pd
import pandas as pd

# To import train_test_split
from sklearn.model_selection import train_test_split

# We can import RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier

# metrics can be imported by using the following line
from sklearn import metrics

# we are importing numpy as np
import numpy as np

# Importing the dataset from the local storage
df = pd.read_csv("/Users/yashbalkhande/Downloads/Crop_Recommdation_Project/Crop_recommendation.csv")

''' Dividing the dataset into dependent and independent variables by using the following lines
features is the independent varible
target is the dependent varible
'''
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Splitting the dataset into training and testing datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

# Making RF as an object of RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100)

# Fitting the Model with the training datasets
RF.fit(Xtrain.values,Ytrain.values)

# We are predicting the output on the basis of the value which is passed in the predict function 
predicted_values = RF.predict(Xtest.values)

'''Accuracy of the model can be checked by comparing the output dataframe with the actual dataframe result
It can be done by using the following lines'''
x = metrics.accuracy_score(Ytest.values, predicted_values)

# printing the accuracy of the model
print("Random Forest Classifier's Accuracy is: ", x*100)


# [83, 45, 60, 28, 70.3, 7.0, 150.9]
# [104,18, 30, 23.603016, 60.3, 6.7, 140.91]

# Creating a numpy array of the desired values
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])

# Predicting the output on the basis of the numpy array which is passed to the predict function
prediction = RF.predict(data)

#printing the output
print(prediction)