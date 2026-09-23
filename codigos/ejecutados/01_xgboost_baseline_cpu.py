#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("data/raw"))


# In[2]:


from sklearn import preprocessing
import xgboost as xgb


# In[3]:


train_transaction = pd.read_csv('data/raw/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('data/raw/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('data/raw/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('data/raw/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('data/raw/sample_submission.csv', index_col='TransactionID')


# In[4]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

# normaliza id-XX -> id_XX (quirk del CSV original de test_identity)
test.columns = [c.replace('id-', 'id_') for c in test.columns]

print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[5]:


del train, test, train_transaction, train_identity, test_transaction, test_identity


# In[6]:


# Label Encoding
for f in X_train.columns:
    if not (pd.api.types.is_numeric_dtype(X_train[f]) and pd.api.types.is_numeric_dtype(X_test[f])):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[7]:


clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999)

clf.fit(X_train, y_train)


# In[8]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')

