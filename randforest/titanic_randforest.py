#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 08:54:35 2020

This script is meant to test out the scikit-learn random forest regression
module. The input data is the Titanic dataset from Kaggle. The goal of this
analysis will be to predict the survival status of a set of passengers.

@author: Alex Arsenault
"""

"""
Import packages
"""
import pandas as pd
import numpy as np
import re, csv

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


"""
Scrub through ticket strings and removing extraneous letters.
"""
def ticket_scrub(df_in):
    
    for i in range(df_in.shape[0]):
        curr_string = df_in["Ticket"].iloc[i]
    
        if (curr_string.isdigit() == True):  # string is already all digits
            temp_string = curr_string
        else:
            for j in range(len(curr_string)): 
                if ((curr_string[j]) == " "): # we've reached a space in string
                    temp_string = curr_string[(j+1):]
                    df_in["Ticket"].iloc[i] = temp_string
    
        if (bool(re.search('[a-zA-Z]', curr_string)) == True): # one last check
            temp_string = "0"
            df_in["Ticket"].iloc[i] = temp_string
        
    # Convert values to int type    
    df_in['Ticket'] = df_in['Ticket'].astype(int)
        
    # Impute '0' values to mean ticket value
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    df_in['Ticket'] = imputer.fit_transform(df_in[['Ticket']])
    
    return df_in    # return modified DataFrame

"""
Scrub through age values.
"""
def age_scrub(df_in):
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    df_in['Age'] = np.nan_to_num(df_in['Age'])
    df_in['Age'] = imputer.fit_transform(df_in[['Age']])
    return df_in

"""
Function for scrubbing through fare values.
"""
def fare_scrub(df_in):
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    df_in['Fare'] = np.nan_to_num(df_in['Fare'])
    df_in['Fare'] = imputer.fit_transform(df_in[['Fare']])
    return df_in

"""
Categorize sex values.
"""
def cat_sex(df_in):
    df_in = pd.get_dummies(df_in, prefix=['Sex'], columns=['Sex'])
    return df_in

"""
Categorize embarked values.
"""
def cat_embark(df_in):
    df_in = pd.get_dummies(df_in, prefix=['Embarked'], columns=['Embarked'])
    return df_in

"""
Categorize Pclass values.
"""
def cat_pclass(df_in):
    df_in = pd.get_dummies(df_in, prefix=['Pclass'], columns=['Pclass'])
    return df_in

"""
Clean our input data.
"""
def clean_df(df_in):

    # Scrub ticket values to remove letters and impute
    df_in = ticket_scrub(df_in)

    # Scrub age values to remove NaN's and impute
    df_in = age_scrub(df_in)
    
    # Scrub fare values to remove NaN's and imput
    df_in = fare_scrub(df_in)
    
    # Encode sex to categorical using pd one-hot
    df_in = cat_sex(df_in)

    # Encode embarked values using pd one-hot    
    df_in = cat_embark(df_in)
    
    # Encode pclass values using pd one-hot    
    df_in = cat_pclass(df_in)
    
    return df_in
    
"""
Select features and drop NaNs.
"""
def feat_select(df_in, *argv):
    feature_list = []
    itr = 0
    
    for itr in range(len(argv[0])):
        feature_list.append(argv[0][itr])
        
    df_in = df_in[feature_list].dropna()
    pass_id = df_in.pop('PassengerId')
    
    return df_in, pass_id 
    

"""
Main entry point for program.
"""
def main():

    # Import training and test data
    train_df = pd.read_csv('../knn/titanic_train.csv')
    test_df = pd.read_csv('../knn/titanic_test.csv')
    
    # Clean input data frames
    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    
    # Feature list 1
    feature_list = ['PassengerId','Age','SibSp','Parch','Ticket','Fare','Sex_female',\
                   'Sex_male','Embarked_C','Embarked_Q','Embarked_S',\
                   'Pclass_1','Pclass_2','Pclass_3']
    
    """
    # Feature list 2
    feature_list = ['Age','SibSp','Parch','Fare','Sex_female',\
                   'Sex_male','Embarked_C','Embarked_Q','Embarked_S',\
                   'Pclass_1','Pclass_2','Pclass_3']
        
    # Feature list 3
    feature_list = ['Age','SibSp','Parch','Fare','Sex_female',\
                   'Sex_male',\
                   'Pclass_1','Pclass_2','Pclass_3']
    """

    feature_list_train = ['Survived'] + feature_list
    
    # Select features for training set
    [train_df, train_pass_id] = feat_select(train_df, feature_list_train)
    
    # Select features for test set
    [test_df, test_pass_id] = feat_select(test_df, feature_list)


    # Pop off target value
    y = train_df.pop('Survived')

    # Initialize a RF Regressor and fit to data
    rf = RandomForestRegressor(n_estimators = 1000, random_state= 23, oob_score=True)
    rf.fit(train_df, y)
    
    # Grab OOB prediction values for each training record
    prediction_pct = rf.oob_prediction_
    prediction_survived = []
    
    # Loop through and determine Survived = 0 or 1
    for i in range(len(prediction_pct)):
        if (prediction_pct[i] >= .5):
            prediction_survived.append(1)
        else:
            prediction_survived.append(0)
            
    # Now grade the OOB survival predictions
    grading_array = [];
    for i in range(len(prediction_survived)):
        if (prediction_survived[i] == y[i]):    # prediction was correct
            grading_array.append(1)
        else:
            grading_array.append(0)             # prediction was incorrect

    # Calculate percentage correct            
    pct_correct = sum(grading_array)/len(grading_array)

    print("Percentage of correct OOB predictions: %f" %(pct_correct))
    
    # Predict the test set
    test_set_prediction_pct = rf.predict(test_df)
    test_set_prediction_survived = []
    
    
    # Loop through and determine Survived = 0 or 1, write to output
    with open ('titanic_results.csv', mode='w') as titanic_results:
        titanic_write = csv.writer(titanic_results,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        titanic_write.writerow(["PassengerId", "Survived"])
        
        for i in range(len(test_set_prediction_pct)):
            
            temp_result = 0
            
            if (test_set_prediction_pct[i] >= .5):
                test_set_prediction_survived.append(1)
                temp_result = 1
            else:
                test_set_prediction_survived.append(0)
                temp_result = 0
            
            titanic_write.writerow([test_pass_id.iloc[i], temp_result])
    
    print("Ending main.")
    
if __name__ == "__main__":
    main()
