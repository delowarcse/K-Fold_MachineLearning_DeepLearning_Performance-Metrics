# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

%matplotlib inline

# Load and merge datasets # white = control; red = stroke; wine = data
No_Concussion = pd.read_csv('S2P_KinarmDataOnly_NoConcussion_byDropingEmpty03.csv', delim_whitespace=False)
Yes_Concussion = pd.read_csv('S2P_KinarmDataOnly_YesConcussion_byDropingEmpty03.csv', delim_whitespace=False)

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

# Libraries for CV, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

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
