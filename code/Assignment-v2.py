#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
import numpy as np
import sklearn

#read data from file
missing_value = ['-9', 'NaN']
data = pd.read_csv(r"D:\Degree\Y2S3\Data Mining\Assignment\Data\Unprocessed_Data02.csv",skipinitialspace=True,na_values=missing_value)


# In[2]:


#create a list to drop the redundant data
drop_columns = ['id','ccf', 'pncaden', 'htn', 'years', 'dm', 'exerwm', 'thalsev','thalpul', 'earlobe','diag','ramus',
               'om2','cathef','junk','restckm','exeref','exerckm','restwm','restef','famhist','name','lvf','lvx1','lvx2','lvx3','lvx4'
               ,'rldv5','ekgmo','ekgyr','ekgday','dig','prop','nitr','pro','diuretic','proto','met','thalrest','tpeakbps'
               ,'dummy','trestbpd','xhypo','rldv5e','cmo','cday','c','lmt','ladprox','laddist','cxmain','om1','rcaprox'
               ,'rcadist','tpeakbpd','painloc','painexer','relrest']

for col in drop_columns:
    if col in data:
        data.drop(col, axis=1, inplace=True)


# In[3]:


#drop duplicate data
data.drop_duplicates(subset=None, keep='first',inplace=True)

#remove outlier
num_columns = data.loc[:, ~data.columns.isin(['sex','cp','smoke','fbs','restecg','exang','ca','thal','num'])]

data_summary = data[num_columns.columns].describe()

quartile_1 = data_summary.loc['25%']
quartile_3 = data_summary.loc['75%']
IQR = quartile_3 - quartile_1
range = 1.5 *IQR

noisy_data =[]
for col in num_columns:
    lower_boundaries = quartile_1[col] - range[col]
    upper_boundaries = quartile_3[col] + range[col]
    noisy_data += data.index[(data[col] < lower_boundaries) | (data[col] > upper_boundaries)].tolist()

data = data.drop(index=noisy_data)


# In[4]:


#check and drop invalid data
sex_query = ((data['sex'] >=3) | (data['sex'] <= -1))
cp_query = ((data['cp']>= 5) | (data['age'] <= 0))
smoke_query = ((data['smoke']>= 2) | (data['smoke'] <= -1))
fbs_query = ((data['fbs']>= 2) | (data['fbs'] <= -1))
restecg_query = ((data['restecg']>= 4) | (data['restecg'] <= -1))
exang_query = ((data['exang']>= 2) | (data['exang'] <= -1))
slope_query = ((data['slope']>= 4) | (data['slope'] <= -1))
ca_query = ((data['ca']>= 4) | (data['ca'] <= -1))
thal_query = ((data['thal'] >= 8) | (data['thal'] <= 2))
num_query = ((data['num'] >= 2) | (data['num'] <= -1))

data = data[~sex_query]
data = data[~cp_query]
data = data[~smoke_query]
data = data[~fbs_query]
data = data[~restecg_query]
data = data[~exang_query]
data = data[~slope_query]
data = data[~ca_query]
data = data[~thal_query]
data = data[~num_query]


# In[5]:


#Fill in data with KNNImputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(data)
data = pd.DataFrame(imputed, columns= data.columns)


# In[6]:


#Bining data(smoke)
label = ['0','1']

enco = preprocessing.LabelEncoder()
data['smoke'] = enco.fit_transform(data['smoke'].astype(str))
data['smoke'] = pd.qcut(data['smoke'], q=2, labels=label, duplicates='drop')


# In[7]:


unormalized_data = data.copy()

#normalized all columns
from sklearn.preprocessing import MinMaxScaler

normalize_columns = ['sex', 'smoke','cigs','fbs','restecg','exang','oldpeak','thal','num']
scaler = MinMaxScaler()
data[normalize_columns] = scaler.fit_transform(data[normalize_columns])


# In[8]:


#Visualize duplicated data with histogram
flg, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,14))
unormalized_data[unormalized_data.columns].hist(column=data.columns, bins=5, ax=ax[0])
ax[0].set_title('Duplicate Rows')
plt.show()


# In[11]:


#Visualize duplicated data with histogram

flg, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,14))
data[data.columns].hist(column=data.columns, bins=5, ax=ax[0])
ax[0].set_title('Duplicate Rows')
plt.show()


# In[ ]:


data.read_csv('')

