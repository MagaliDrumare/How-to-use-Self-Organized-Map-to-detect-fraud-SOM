#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:45:22 2017
Credit to :  Deep Learning A to Z Udemy 
"""

#UNSUPERVISED MODEL 
# Import the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Make Customer segmentation : identify segment of customers. 
# Inputs are the customers and Output is a map of theses customers. 
# Each zone of the map group customers with same patterns. 
# Fraud customers will be made of one specific segment of the SOM. 
# Step 1 : Determine the winning node for each rows of the dataset (the customers)
# Step 2 : Use a neighborhood function to update the weights 
# of the neighbors node to move these nodes closest to the winning node. 
# Step 3 : We do this for all the customers and repeat it. 
# and reach a time the neighborhood stops decreasing.  
# and then we obtain the final mapping. 


## Part 1 : Prepare the data 

# Importing the dataset
dataset=pd.read_csv("Credit_Card_Applications.csv")

# We only use X to make the SOM 
X=dataset.iloc[:,:-1].values
#Customers approved or nor approved. 
y=dataset.iloc[:,-1].values

#Feature scaling, all the data between 0 and 1 
from sklearn.preprocessing import MinMaxScaler 
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)


## Part 2 : Create the data and  train in on the dataset X 

#Training the SOM (with MiniSom 1.0-minisom.py)
#Import the MiniSom class from minisom.py 
from minisom import MiniSom 

#Create a 10x10 grid composed of nodes 
#15 features in X
#sigma = radius of the neighborhood 
#Create the object som from the cass MiniSom
som=MiniSom(x=10,y=10,input_len=15, sigma=1.0, learning_rate=0.5) 

#Initialization of the SOM weights 
som.random_weights_init(X)
#Train the SOM on X 
som.train_random(data=X,num_iteration=100)


## PART 3 : Visualize the result (plot the SOM). 
#Two dimensional grid. 
#How to detect the outlier. 
#Find the winning node that has the highest MID  (mean into neuron distance.) 
from pylab import bone, pcolor, colorbar, plot, show 
bone()
# A method of MiniSom class that take all the MID of the winning nodes
pcolor(som.distance_map().T)
#Larger is the MID ==white;Smaller is the MID ==black 
#We obtain the normalize data between 0-1
colorbar()

#Add some maker to associate the winning node 
#With approved or non approved customers 
markers=['o','s']
colors= ['r','g']
# i is the index ans x the customer
for i,x in enumerate(X):
#Get the winning node of the customer x 
# Use a method winner of the class som 
# To obtain the coordinates of the winning nodes
    w=som.winner(x)
# w[0]and w[1] are the two coordinates of the winning nodes. 
# markers[y[i]] is Round or Square for approved or not.
# markeredgecolor=colors[y[i]] green or red
# markerfacecolor= 'None' empty circle  
    plot(w[0]+0.5,w[1]+0.5, 
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor= 'None',
         markersize=10,
         markeredgewidth=1)
    
## PART 4 Find the frauds 
# number of customers associated to the winning nodes. 
mappings=som.win_map(X)  

# lists associated with the outlier winning node 
frauds = mappings[(3,5)]
#frauds = np.concatenate((mappings[(5,4)], mappings[(4,1)]), axis = 0)
# inverse the normalization of the data. 
frauds = sc.inverse_transform(frauds)



    
    


 



