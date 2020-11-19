# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:46:07 2018

@author: doganay
"""

import numpy as np
import pandas as pd
import math


#Importing the datasets

dataset = pd.read_csv("hw01_data_set_images.csv",header = None)
dataset1 = pd.read_csv("hw01_data_set_labels.csv",header = None)

X = dataset.iloc[:,:].values
y = dataset1.iloc[:,:].values


#Setting the train and test splits

train_X = np.concatenate((X[0:25],X[39:64],X[78:103],X[117:142],X[156:181]))
train_y = np.concatenate((y[0:25],y[39:64],y[78:103],y[117:142],y[156:181]))
test_X = np.concatenate((X[25:39],X[64:78],X[103:117],X[142:156],X[181:195]))
test_y = np.concatenate((y[25:39],y[64:78],y[103:117],y[142:156],y[181:195]))


#Calculation of pcd values

pcd=list()
toplam = np.zeros([5, 320], dtype = int) 
for d in range(0,320):
    for i in range(0,125):
        if (train_y[i]=='A'):
            class_label=0
            toplam[class_label][d]=toplam[class_label][d]+train_X[i][d]
        if (train_y[i]=='B'):
            class_label=1
            toplam[class_label][d]=toplam[class_label][d]+train_X[i][d]
        if (train_y[i]=='C'):
            class_label=2
            toplam[class_label][d]=toplam[class_label][d]+train_X[i][d]
        if (train_y[i]=='D'):
            class_label=3
            toplam[class_label][d]=toplam[class_label][d]+train_X[i][d]
        if (train_y[i]=='E'):
            class_label=4
            toplam[class_label][d]=toplam[class_label][d]+train_X[i][d]

pcd=toplam/25



#Calculation of scoring function "gc" 
gc = np.zeros([5, 125], dtype = float) 

for i in range (0,125):
    for d in range (0,320):
        for k in range(0,5):
            if (pcd[k][d]==0 or pcd[k][d]==1):
                gc[k][i]+=0
            else:
                gc[k][i] += train_X[i][d]*math.log(pcd[k][d])+(1-train_X[i][d])*math.log(1-pcd[k][d])

#Estimation of train test labels
                
y_hat_train = np.zeros(125, dtype=int) 
for i in range(0,125):
    if max(gc[:,i])==gc[0,i]:
        y_hat_train[i]=1
    if max(gc[:,i])==gc[1,i]:
        y_hat_train[i]=2
    if max(gc[:,i])==gc[2,i]:
        y_hat_train[i]=3
    if max(gc[:,i])==gc[3,i]:
        y_hat_train[i]=4
    if max(gc[:,i])==gc[4,i]:
        y_hat_train[i]=5

#Changing the letters to numbers for confusion matrix calculations
y_train_number = np.zeros(125, dtype=int) 

for i in range(0,125):
    if train_y[i]=='A':
        y_train_number[i]=1
    if train_y[i]=='B':
        y_train_number[i]=2
    if train_y[i]=='C':
        y_train_number[i]=3
    if train_y[i]=='D':
        y_train_number[i]=4
    if train_y[i]=='E':
        y_train_number[i]=5


#Confusion Matrix for Training Set
        
confusion_matrix = np.zeros([5,5], dtype=int)

for i in range(0,125):
    for j in range(1,6):
        for k in range(1,6):
            if y_hat_train[i]==j and y_train_number[i]==k:
                confusion_matrix[(j-1),(k-1)]+=1


#Scoring Function calculation for Test Set

gc_test = np.zeros([5, 70], dtype = float) 

for i in range (0,70):
    for d in range (0,320):
        for k in range(0,5):
            if (pcd[k][d]==0 or pcd[k][d]==1):
                gc_test[k][i]+=0
            else:
                gc_test[k][i] += test_X[i][d]*math.log(pcd[k][d])+(1-test_X[i][d])*math.log(1-pcd[k][d])

#Estimation of test test labels

y_hat_test = np.zeros(70, dtype=int) 
for i in range(0,70):
    if max(gc_test[:,i])==gc_test[0,i]:
        y_hat_test[i]=1
    if max(gc_test[:,i])==gc_test[1,i]:
        y_hat_test[i]=2
    if max(gc_test[:,i])==gc_test[2,i]:
        y_hat_test[i]=3
    if max(gc_test[:,i])==gc_test[3,i]:
        y_hat_test[i]=4
    if max(gc_test[:,i])==gc_test[4,i]:
        y_hat_test[i]=5
        
#Changing the letters to numbers for confusion matrix calculations

y_test_number = np.zeros(70, dtype=int) 

for i in range(0,70):
    if test_y[i]=='A':
        y_test_number[i]=1
    if test_y[i]=='B':
        y_test_number[i]=2
    if test_y[i]=='C':
        y_test_number[i]=3
    if test_y[i]=='D':
        y_test_number[i]=4
    if test_y[i]=='E':
        y_test_number[i]=5
        
#Confusion Matrix for Test Set

confusion_matrix_test = np.zeros([5,5], dtype=int)

for i in range(0,70):
    for j in range(1,6):
        for k in range(1,6):
            if y_hat_test[i]==j and y_test_number[i]==k:
                confusion_matrix_test[(j-1),(k-1)]+=1
