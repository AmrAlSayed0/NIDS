# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:02:20 2021

@author: mahmoud
"""


# Import the required libraries
import os
import math
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load features names
#%%
features_names = pd.read_csv('kddcup.names')
header= features_names['back'].str.split(expand=True)
features = np.array(header[0])
features = [features[i].rstrip(':') for i in range(len(features))]


#load the data set
#with features' names added
df = pd.read_csv ( 'corrected',names=features )
#%%



# Inspect the data type of each feature. Maybe convert them to more appropriate
# data type later.
num_of_data_points = df.shape [ 0 ]
num_of_features = df.shape [ 1 ]
df.info ( verbose = True )

# Looks like there are data that are misrepresented as 'object' or 'int64' 
#when they're in fact category strings or booleans
# Here we fix that
#print(df.head)

df [ 'protocol_type' ] = df [ 'protocol_type' ].astype ( 'category' )
df [ 'service' ] = df [ 'service' ].astype ( 'category' )
df [ 'flag' ] = df [ 'flag' ].astype ( 'category' )
df [ 'land' ] = df [ 'land' ].astype ( 'category' )
df [ 'logged_in' ] = df [ 'logged_in' ].astype ( 'category' )
df [ 'is_host_login' ] = df [ 'is_host_login' ].astype ( 'category' )
df [ 'is_guest_login' ] = df [ 'is_guest_login' ].astype ( 'category' )
df [ 'target' ] = df [ 'dst_host_srv_rerror_rate' ].astype ( 'category' )
del df['dst_host_srv_rerror_rate']


# Look and check the conversion is correct
df.info ( verbose = True )

# Check the data for any unusual or invalid values (e.g negative values for duration or byte size or count and values above 1 for rate)
df.describe ()

# Quick scan for categorical columns
df.describe ( include = 'category' )

# Check if there are any nan or invalid values. Remove them if there is.
df [ df.isna ().any ( axis = 1 ) ]

df [ 'protocol_type' ].unique ()
df [ 'service' ].unique ()
df [ 'flag' ].unique ()
df [ 'land' ].unique ()
df [ 'logged_in' ].unique ()
df [ 'is_host_login' ].unique ()
df [ 'is_guest_login' ].unique ()
df [ 'target' ].unique ()

# First, it was notied that the feature named 'num_outbound_cmds' is nothing but zero values.
df [ 'num_outbound_cmds' ].describe ()

# So we drop it
df = df.drop ( columns = [ 'num_outbound_cmds' ] )

# We have a total of 34 numerical/continuous features and 7 categorical features
# We define feature selection methods
x_numerical = df.select_dtypes ( exclude = [ object , 'category' ] )
# Let's say we only need a set percentage of the total number of features
percentage_of_features = 0.5
num_of_numerical_features = x_numerical.shape [ 1 ]
num_of_selected_numerical_features = math.ceil ( num_of_numerical_features * percentage_of_features )
# Here we use Analysis Of Variance (AVONA) F-Test. It is best suited for numerical input and categorical output.
fs = SelectKBest ( score_func = f_classif , k = num_of_selected_numerical_features )
# Apply the feature selection
y = df [ 'target' ]
x_numerical_selected = fs.fit_transform ( x_numerical , y )
x_numerical.loc [ : , fs.get_support ( indices = False ) ]

corr = df.select_dtypes ( exclude = [ object , 'category' ] ).corr (method='pearson')
# Generate a mask for the upper triangle
mask = np.triu ( np.ones_like ( corr , dtype = bool ) )

# Set up the matplotlib figure
f , ax = plt.subplots ( figsize = (20 , 20) )

# Generate a custom diverging colormap
cmap = sns.diverging_palette ( 240 , 360 , as_cmap = True )

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap ( corr , mask = mask , cmap = cmap , vmax = .3 , center = 0 , square = True , linewidths = .5 , cbar_kws = {
    "shrink" : .5
    } )

y = np.array(df[['target']])



#%%
def catMultiLabels(arr):
    for k in range(len(arr)):
        if arr[k]=='normal.':
            arr[k]=0
        elif arr[k]=='snmpgetattack.':
            arr[k]=1
        elif arr[k]=='smurf.':
            arr[k]=2
        elif arr[k]=='neptune.':
            arr[k]=3
        else:
            arr[k]=4
    return arr

def catBinaryLabels(arr):
    for k in range(len(arr)):
        if arr[k]=='normal.':
            arr[k]=0
        else:
            arr[k]=1
    return arr

y = catMultiLabels(y).astype('int')

df['target']=y
df [ 'target' ] = df [ 'target' ].astype ( 'int' )
#%%
## Split data

X_train,X_test,y_train,y_test = train_test_split(x_numerical_selected,y,test_size=0.2,random_state=1)

#%%
# KNN Algorithm
print("KNN Algorithm")
Le = LabelEncoder()
for i in range(len(X_train[0])):
    X_train[:,i]=Le.fit_transform(X_train[:,i])
for i in range(len(X_test[0])):
    X_test[:,i]=Le.fit_transform(X_test[:,i])

knn = neighbors.KNeighborsClassifier(n_neighbors=5,weights='uniform')
print("Training ...")
knn.fit(X_train,y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("Predictions",prediction)
print("Accuracy",accuracy)

#%% random forest
X_train,X_test,y_train,y_test = train_test_split(x_numerical_selected,y,test_size=0.2,random_state=1)
print("Random forest")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = RandomForestClassifier(n_estimators = 5,random_state=0)
print("Training")
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_predict)
print("Predictions",prediction)
print("Accuracy",accuracy)
