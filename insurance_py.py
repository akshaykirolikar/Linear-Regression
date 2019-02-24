# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:11:01 2019

@author: Akshay
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pl
import seaborn as sns

data = pd.read_csv('insurance.csv')

#checking for missing values
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)

#correation between variables
data.corr()['charges'].sort_values()

pl.matshow(data.corr())


f= pl.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.smoker == 1)]["charges"],color='c',ax=ax)
ax.set_title('Distribution of charges for smokers')

ax=f.add_subplot(122)
sns.distplot(data[(data.smoker == 0)]['charges'],color='b',ax=ax)
ax.set_title('Distribution of charges for non-smokers')

X=data.iloc[:,:-1]

y=data[['charges']]

from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
x_scaled = std_scaler.fit_transform(X)
x_scaled= pd.DataFrame(x_scaled)

#correation matrix
p=data.corr()


# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1338, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5,6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_1=X[:,[1,3,4,5]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size = 0.20, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#y_pred vs y_test
pl.style.use('ggplot')
pl.scatter( y_test, y_pred, color = "green")
pl.xlabel('Charges_Observed', fontsize=18)
pl.ylabel('Charges_Predicted', fontsize=16)
pl.show()


    
