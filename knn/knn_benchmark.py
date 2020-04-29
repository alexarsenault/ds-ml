#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:13:28 2020

@author: alex
Python Knn Benchmark
"""


"""
Import packages
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, csv
import time
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#@profile
def run_knn():
    
    num_rows = 10000000

    # Generate random values for fake test data
    rand_lat = (np.random.random(num_rows) - 0.5)*(90/.5)
    rand_lon = (np.random.random(num_rows) - 0.5)*(180/.5)
    rand_rf = (np.random.rand(num_rows)*1000) + 2500
    rand_occur = (np.random.randint(low = 0, high = 3, size = num_rows))
    rand_pct = (np.random.random(num_rows)*5) + 95
    rand_num1 = (np.random.random(num_rows)*100)
    rand_num2 = (np.random.random(num_rows)*100)
    rand_num3 = (np.random.random(num_rows)*100)
    rand_num4 = (np.random.random(num_rows)*100)
    rand_num5 = (np.random.random(num_rows)*100)
    rand_num6 = (np.random.random(num_rows)*100)
    rand_num7 = (np.random.random(num_rows)*100)
    rand_num8 = (np.random.random(num_rows)*100)
    rand_num9 = (np.random.random(num_rows)*100)
    rand_num10 = (np.random.random(num_rows)*100)
    rand_num11 = (np.random.random(num_rows)*100)
    rand_num12 = (np.random.random(num_rows)*100)
    rand_num13 = (np.random.random(num_rows)*100)
    rand_num14 = (np.random.random(num_rows)*100)
    rand_num15 = (np.random.random(num_rows)*100)
    rand_num16 = (np.random.random(num_rows)*100)
    rand_num17 = (np.random.random(num_rows)*100)
    rand_num18 = (np.random.random(num_rows)*100)
    rand_num19 = (np.random.random(num_rows)*100)
    rand_num20 = (np.random.random(num_rows)*100)

    rand_target = (np.random.random(num_rows))

    rand_data_set = np.column_stack((rand_lat,rand_lon,rand_rf,rand_occur,rand_pct,
                                 rand_num1, rand_num2, rand_num3, rand_num4, 
                                 rand_num5, rand_num6, rand_num7, rand_num8, 
                                 rand_num9, rand_num10, rand_num11, rand_num12, 
                                 rand_num13, rand_num14, rand_num15, rand_num16, 
                                 rand_num17, rand_num18, rand_num19, rand_num20,))

    x = rand_data_set
    y = rand_target

    X_train_classify, X_test_classify, y_train_classify, y_test_classify = train_test_split(x,y, test_size = 0.2, random_state=42)
    rand_data_train, rand_data_test, y_data_train, y_data_test = train_test_split(rand_data_set, rand_pct, test_size = 0.2, random_state=42)

    print("X_train_classify")
    print(type(X_train_classify))

    neighbors = np.arange(50,100,50)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    train_accuracy_regressor = np.empty(len(neighbors))
    test_accuracy_regressor = np.empty(len(neighbors))

    
    
    for i, k in enumerate(neighbors):
        print(k)
    
        start_time = time.thread_time_ns()
    
        knn_regressor = KNeighborsRegressor(n_neighbors=10000,algorithm="kd_tree")
        knn_regressor.fit(rand_data_train, y_data_train)
        
        end_time = time.thread_time_ns()
        model_time = end_time - start_time
        print("k model: " + str(model_time*(1e-9)) + " s.")




if __name__ == '__main__':
    run_knn()


