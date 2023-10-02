#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import pyplot
import xgboost
from xgboost import XGBRegressor
import sys
import scipy
import sklearn
import itertools # construct specialized tools
from matplotlib import rcParams # plot size customization
from termcolor import colored as cl # text customization
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv


# In[13]:


df = pd.read_csv("TrainingSetBTCChallenge.csv")
df.head()


# In[14]:


df.tail()


# In[15]:


import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(41,41))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[16]:


df.info()


# In[17]:


df.shape


# In[18]:


df['TARGET'].value_counts()


# In[19]:


df.describe()


# In[20]:


y = df.iloc[:,0:1]


# In[21]:


y


# In[22]:


X = df.iloc[:,1:]


# In[23]:


X


# In[11]:


cor = df.corr()
pd.plotting.scatter_matrix(cor,figsize = (30,30), diagonal = 'kde')
pyplot.show()


# In[24]:


train_test_split(X,y,test_size = 0.15, random_state=1)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15, random_state=1)


# In[47]:


xgbR=xgboost.XGBRegressor()
xgbR.fit(X_train, y_train)


# In[28]:


score = xgbR.score(X_train, y_train)
print(score)


# In[29]:


cv_score = cross_val_score(xgbR, X_train, y_train, cv = 10)
print("cv score:", cv_score)
print("cv mean score:",cv_score.mean())


# In[30]:


y_pred = xgbR.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
print("MSE:",MSE)
print("RMSE:",np.sqrt(MSE))


# In[33]:


y_pred


# In[35]:


from sklearn.metrics import r2_score
print(r2_score(y_pred, y_test))


# In[44]:


model = RandomForestClassifier(n_estimators=100)
# Fit on training data
model.fit(X_train, y_train)


# In[37]:


def test_data(real, pre):    
    total = 0

    for i in range(0, len(pre)):
        d = abs(pre[i] - real[i])
        print(f"{pre[i]} - {real[i]} = {d}")

        total += d
    
    print(total/len(pre))


# In[57]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[49]:


y_test.values()


# In[51]:


y_panda = pd.DataFrame(y_pred, columns =["TOTAL"])


# In[52]:


y_panda.hist()
plt.show()


# In[56]:


y_panda.describe()


# In[73]:


y_train = np.ravel(y_train)


# In[74]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[75]:


lr = LogisticRegression(C = 0.1, solver = 'liblinear')
lr.fit(X_train,y_train)


# In[ ]:




