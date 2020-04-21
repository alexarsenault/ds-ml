#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:21:52 2020

@author: alex
"""

"""
Import packages
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    #df_in = pd.get_dummies(df_in, prefix=['Sex'], columns=['Sex'])
    df_in["Sex"] = df_in["Sex"].astype('category')
    df_in["Sex"] = df_in["Sex"].cat.codes
    return df_in

"""
Categorize embarked values.
"""
def cat_embark(df_in):
    #df_in = pd.get_dummies(df_in, prefix=['Embarked'], columns=['Embarked'])
    df_in["Embarked"] = df_in["Embarked"].astype('category')
    df_in["Embarked"] = df_in["Embarked"].cat.codes
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
    #df_in = cat_pclass(df_in)
    
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
    train_df = pd.read_csv('../data/titanic_data/titanic_train.csv')
    test_df = pd.read_csv('../data/titanic_data/titanic_test.csv')
    
    # Clean input data frames
    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    
    # Feature list 1
    """
    feature_list = ['PassengerId','Age','SibSp','Parch','Ticket','Fare','Sex_female',\
                   'Sex_male','Embarked_C','Embarked_Q','Embarked_S',\
                   'Pclass_1','Pclass_2','Pclass_3']
    
    # Feature list 2
    feature_list = ['PassengerId','Age','SibSp','Parch','Fare','Sex_female',\
                   'Sex_male','Embarked_C','Embarked_Q','Embarked_S',\
                   'Pclass_1','Pclass_2','Pclass_3']
        
    # Feature list 3
    feature_list = ['PassengerId','Age','SibSp','Parch','Fare','Sex_female',\
                   'Sex_male','Pclass_1','Pclass_2','Pclass_3']
    
    # Feature list 4
    feature_list = ['PassengerId','Age','Fare','Sex_female',\
                   'Sex_male','Pclass_1','Pclass_2','Pclass_3']
    """

    # Feature list 5 (no one-hot encoding for PClass / Sex)
    feature_list = ['Pclass','Age','SibSp','Parch','Fare','Sex','PassengerId']
    
    feature_list_train = ['Survived'] + feature_list
    
    # Select features for training/test set
    [train_df, train_pass_id] = feat_select(train_df, feature_list_train)
    [test_df, test_pass_id] = feat_select(test_df, feature_list)

    # Pop off target value of training set
    y = train_df.pop('Survived')

    # Partition training and test data
    data_train, data_test, y_train, y_test = train_test_split(train_df, y, \
                                            test_size = 0.2, random_state=21)

    # Loop through and try multiple values of K
    num_trials = 50
    score_array = np.zeros(num_trials)
    for k in range(1,num_trials):
        # Initialize a RF Regressor and fit to data
        knn = KNeighborsRegressor(n_neighbors = k, metric = 'euclidean')
        knn.fit(data_train, y_train)
        score_array[k] = knn.score(data_test, y_test)
  
    # Determine optimal k
    k = score_array.argmax()
    print("Best K to use is: " + str(k))  
  
    # Plot scores wrt k
    plt.scatter(range(1,num_trials),score_array[1:])
  
    # Final determination for model
    knn = KNeighborsRegressor(n_neighbors = k, metric = 'euclidean')
    knn.fit(train_df, y)
    final_results = knn.predict(test_df)    

    # Write prediction results
    with open('titanic_results.csv', mode='w') as titanic_results:
        titanic_write = csv.writer(titanic_results,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
        titanic_write.writerow(["PassengerId", "Survived"])
    
        for i in range(0,len(final_results)):
            if(final_results[i] >= .5):
                passenger_result = 1
            else:
                passenger_result = 0
            
            titanic_write.writerow([test_pass_id[i], passenger_result])
    
    print("Ending main.")
    
if __name__ == "__main__":
    main()