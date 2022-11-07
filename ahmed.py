# -*- coding: utf-8 -*-
#### our predictions(Target-Labels) ==> whether customer is buy or not - (binary-classification)

## EDA

import numpy as np
import pandas as pd

# import csv data.
df = pd.read_csv('storepurchasedata.csv')

# Split data --> dat-Features(X), data-Target_Labels(y)
X= df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# Split X, y --> Train/Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=(0))

# Scale-Features(X)
from sklearn.preprocessing import StandardScaler

# Create Object Scalar.
sc = StandardScaler()

# Scale data-Features(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




## Modeling

# Select-Model.
from sklearn.neighbors import KNeighborsClassifier

# minkowski is for ecledian_distance.
classifier =KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Train-Model.
classifier.fit(X_train, y_train)

# Predict on test_data
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

# Evaluate-Model With metrics
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report 

cac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred) 
cr = classification_report(y_test, y_pred)





## Finaly

# Predict1 on new-data
new_pred1 = classifier.predict(sc.transform(np.array([[40,20000]])))

# Predict1-Proba on new-data.
new_pred1_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

# Predict2 on new-data
new_pred2 = classifier.predict(sc.transform(np.array([[42,50000]])))

# Predict2-Proba on new-data.
new_pred2_proba = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]



## Save - Model and Scaler to reuse later.


## Serialize data.
import pickle

model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))

scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))





## DeSerialize data.
import pickle

local_Classifier = pickle.load(open('classifier.pickle', 'rb'))
local_Scaler = pickle.load(open('sc.pickle', 'rb'))


new_pred = local_Classifier.predict(local_Scaler.transform(np.array([[40,20000]])))

new_pred_proba = local_Classifier.predict_proba(local_Scaler.transform(np.array([[40,20000]])))[:,1]

newPred = local_Classifier.predict(local_Scaler.transform(np.array([[42,50000]])))

newPred_proba = local_Classifier.predict_proba(local_Scaler.transform(np.array([[42,50000]])))[:,1]