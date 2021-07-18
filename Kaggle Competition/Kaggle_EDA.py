#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px 
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r'C:\Users\marin\Downloads\train.csv (1).zip')

df.head()


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


display(df.info())


# In[6]:


missing_data = df.isnull().sum(axis=0)
print(missing_data.value_counts())


# In[7]:


duplicated_data = df.duplicated(keep=False)
print(duplicated_data.value_counts())


# In[8]:


unique_values = df.nunique()
print(unique_values)


# In[11]:


print(df['target'].value_counts())
print('\n')
plt.figure(figsize=(10,12))
sns.countplot(x = 'target', data= df, palette="Oranges_r")
plt.show();


# Class_6 and Class_8 | Class_5 and Class_4 

# In[12]:


fig, ax= plt.subplots(figsize=(10,7))
ax=sns.heatmap(df.corr(), cmap="Oranges");


# In[15]:


columns_pairplot = ['feature_0', 'feature_1', 'feature_2']
sns.pairplot(df[columns_pairplot], diag_kind='kde');


# In[ ]:




