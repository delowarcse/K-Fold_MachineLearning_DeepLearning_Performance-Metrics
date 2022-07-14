#!/usr/bin/env python
# coding: utf-8

# # This program is for geenerating performance metrices (accuracy, precision, recall, f1-score)

# In[ ]:


# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets # white = control; red = stroke; wine = data
No_Concussion = pd.read_csv('Helathy Participants Data.csv', delim_whitespace=False)
Yes_Concussion = pd.read_csv('Injured Participant Data.csv', delim_whitespace=False)

# store wine type as an attribute
No_Concussion['data_type'] = 'NoConcussion'   
Yes_Concussion['data_type'] = 'Concussion'

# merge control and stroke data
datas = pd.concat([No_Concussion, Yes_Concussion])
#datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare Training and Testing Datasets
stp_features = datas.iloc[:,:-1]
stp_feature_names = stp_features.columns
stp_class_labels = np.array(datas['data_type'])

X_data = datas.iloc[:,:-1]
y_label = datas.iloc[:,-1]

# Data Normalization
ss = StandardScaler().fit(X_data)
X = ss.transform(X_data)
le = LabelEncoder()
le.fit(y_label)
y = le.transform(y_label)


# In[ ]:


# Libraries for CV, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}


# In[ ]:


from sklearn.linear_model import LogisticRegression

# create model
model_lr = LogisticRegression()

results_lr = cross_validate(estimator=model_lr,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_lr)

print('Logistic Regression Metrics:')
print('Fit time:',results_lr['fit_time'])
print('Score time:',results_lr['score_time'])
print('10-fold Accuracy:',results_lr['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_lr['test_accuracy']),np.std(results_lr['test_accuracy'])))
print('10-fold Precision:',results_lr['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_lr['test_precision']),np.std(results_lr['test_precision'])))
print('10-fold Recall:',results_lr['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_lr['test_recall']),np.std(results_lr['test_recall'])))
print('10-fold f1-score:',results_lr['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_lr['test_f1_score']),np.std(results_lr['test_f1_score'])))


# In[ ]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

#train model
model_dt = DecisionTreeClassifier(max_depth=4)

results_dt = cross_validate(estimator=model_dt,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_dt)

print('Decision Tree Metrics:')
print('Fit time:',results_dt['fit_time'])
print('Score time:',results_dt['score_time'])
print('10-fold Accuracy:',results_dt['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dt['test_accuracy']),np.std(results_dt['test_accuracy'])))
print('10-fold Precision:',results_dt['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dt['test_precision']),np.std(results_dt['test_precision'])))
print('10-fold Recall:',results_dt['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dt['test_recall']),np.std(results_dt['test_recall'])))
print('10-fold f1-score:',results_dt['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dt['test_f1_score']),np.std(results_dt['test_f1_score'])))


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
model_rf=RandomForestClassifier(n_estimators=50) 

results_rf = cross_validate(estimator=model_rf,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_rf)

print('Random Forest Metrics:')
print('Fit time:',results_rf['fit_time'])
print('Score time:',results_rf['score_time'])
print('10-fold Accuracy:',results_rf['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_rf['test_accuracy']),np.std(results_rf['test_accuracy'])))
print('10-fold Precision:',results_rf['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_rf['test_precision']),np.std(results_rf['test_precision'])))
print('10-fold Recall:',results_rf['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_rf['test_recall']),np.std(results_rf['test_recall'])))
print('10-fold f1-score:',results_rf['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_rf['test_f1_score']),np.std(results_rf['test_f1_score'])))


# In[ ]:


# train the Random Forest Hyperparameter Tuning
model_rft = RandomForestClassifier(n_estimators=200, max_features='auto')

results_rft = cross_validate(estimator=model_rft,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_rft)

print('Random Forest with Hyperparameter Tuning Metrics:')
print('Fit time:',results_rft['fit_time'])
print('Score time:',results_rft['score_time'])
print('10-fold Accuracy:',results_rft['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_rft['test_accuracy']),np.std(results_rft['test_accuracy'])))
print('10-fold Precision:',results_rft['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_rft['test_precision']),np.std(results_rft['test_precision'])))
print('10-fold Recall:',results_rft['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_rft['test_recall']),np.std(results_rft['test_recall'])))
print('10-fold f1-score:',results_rft['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_rft['test_f1_score']),np.std(results_rft['test_f1_score'])))


# In[ ]:


# Support Vector Machine
from sklearn.svm import SVC

model_svm = SVC(random_state=42)

results_svm = cross_validate(estimator=model_svm,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_svm)

print('Support Vector Machine Metrics:')
print('Fit time:',results_svm['fit_time'])
print('Score time:',results_svm['score_time'])
print('10-fold Accuracy:',results_svm['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_svm['test_accuracy']),np.std(results_svm['test_accuracy'])))
print('10-fold Precision:',results_svm['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_svm['test_precision']),np.std(results_svm['test_precision'])))
print('10-fold Recall:',results_svm['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_svm['test_recall']),np.std(results_svm['test_recall'])))
print('10-fold f1-score:',results_svm['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_svm['test_f1_score']),np.std(results_svm['test_f1_score'])))


# In[ ]:


# K-Nearest Neighbors Model

from sklearn.neighbors import KNeighborsClassifier

model_KNN = KNeighborsClassifier(n_neighbors=5)

results_KNN = cross_validate(estimator=model_KNN,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_KNN)

print('K-nearest Neighbors Metrics:')
print('Fit time:',results_KNN['fit_time'])
print('Score time:',results_KNN['score_time'])
print('10-fold Accuracy:',results_KNN['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_KNN['test_accuracy']),np.std(results_KNN['test_accuracy'])))
print('10-fold Precision:',results_KNN['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_KNN['test_precision']),np.std(results_KNN['test_precision'])))
print('10-fold Recall:',results_KNN['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_KNN['test_recall']),np.std(results_KNN['test_recall'])))
print('10-fold f1-score:',results_KNN['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_KNN['test_f1_score']),np.std(results_KNN['test_f1_score'])))


# In[ ]:


# Deep Neural Network
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def create_network():
    # network defination
    network = models.Sequential()
    # Add fully connected layer with a ReLU
    network.add(layers.Dense(units=16, activation='relu', input_shape=(79,)))
    network.add(layers.Dense(units=16, activation='relu'))
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # compile network
    network.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    return network

# wrap keras model
neural_network = KerasClassifier(build_fn=create_network,
                                epochs=100,
                                batch_size=10,
                                verbose=1)

results_dnn = cross_validate(estimator=neural_network,
                          X=X,#X=features,
                          y=y,#y=labels,
                          cv=kfold,
                          scoring=scoring)
print(results_dnn)


# In[ ]:


print('Deep Neural Network Metrics:')
print('Fit time:',results_dnn['fit_time'])
print('Score time:',results_dnn['score_time'])
print('10-fold Accuracy:',results_dnn['test_accuracy'])
print('Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dnn['test_accuracy']),np.std(results_dnn['test_accuracy'])))
print('10-fold Precision:',results_dnn['test_precision'])
print('Precision(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dnn['test_precision']),np.std(results_dnn['test_precision'])))
print('10-fold Recall:',results_dnn['test_recall'])
print('Recall(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dnn['test_recall']),np.std(results_dnn['test_recall'])))
print('10-fold f1-score:',results_dnn['test_f1_score'])
print('f1-score(Mean (Standard Deviation): %f (%f)'%(np.mean(results_dnn['test_f1_score']),np.std(results_dnn['test_f1_score'])))


# In[ ]:


print('Logistic Regression: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_lr['test_accuracy']),np.std(results_lr['test_accuracy'])))
print('Decision Tree: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dt['test_accuracy']),np.std(results_dt['test_accuracy'])))
print('Random Forest: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_rf['test_accuracy']),np.std(results_rf['test_accuracy'])))
print('Random Forest with Hyperparameters Tuning: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_rft['test_accuracy']),np.std(results_rft['test_accuracy'])))
print('Support Vector Machine: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_svm['test_accuracy']),np.std(results_svm['test_accuracy'])))
print('K-nearest Neighbors: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_KNN['test_accuracy']),np.std(results_KNN['test_accuracy'])))
print('Deep Neural Network: Accuracy(Mean (Standard Deviation)): %f (%f)'%(np.mean(results_dnn['test_accuracy']),np.std(results_dnn['test_accuracy'])))


# In[ ]:




