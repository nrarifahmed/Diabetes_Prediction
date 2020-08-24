# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:32:53 2020

@author: Arif Ahmed N R nrarifahmed@gmail.com
### Diabetes Prediction
"""

import numpy as np
import pandas as pd

diabetes_df = pd.read_csv('D:/pycaret_code/diabetes_pycaret.csv') 

diabetes_df

diabetes_df.head()

X = diabetes_df.iloc[:,:-1]
y = diabetes_df.iloc[:,-1]

X
y

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


from sklearn.ensemble import RandomForestClassifier
diabetes_classifier=RandomForestClassifier()
diabetes_classifier.fit(X_train,y_train)

y_pred = diabetes_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)

score

import pickle
pickle_out=open("diabetes_classifier.pkl","wb")
pickle.dump(diabetes_classifier,pickle_out)
pickle_out.close()

