#!/usr/bin/env python
# coding: utf-8

# **TABULAR PLAYGROUND SERIES _ EDA**

# *Importing the principle libraries*

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px 
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# *Importing the dataset from kaggle and starting the EDA valuation*

# In[3]:


df = pd.read_csv(r'C:\Users\marin\Downloads\train.csv.zip')

df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# *Founding missing values*

# In[7]:


missing_data = df.isnull().sum(axis=0)
print(missing_data.value_counts())


# *Founding duplicated values*

# In[8]:


duplicated_data = df.duplicated(keep=False)
print(duplicated_data.value_counts())


# *Rapresentation of data with pairplot for understanding the possible correlation between features*

# In[8]:


sns.pairplot(df, palette="husl");


# *Rappresentation of data with heatmap for a better understanding of the correlation*

# In[31]:


fig, ax= plt.subplots(figsize=(10,7))
ax=sns.heatmap(df.corr(), cmap="Reds", annot=True);


# In[10]:


df_corr = df.corr()['target'][:-1]
list = df_corr[abs(df_corr) > 0.7].sort_values(ascending=False)
print("There are {} strongly correlated values with Target\n{}".format(len(list), list))


# In[11]:


df_corr


# *Founding outliers with the rappresentation of boxplot*

# In[34]:


for i in ["id", "cont0", "cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7", "cont8", "cont9", "cont10"]:
    print(i.upper())
    sns.boxplot(data =df, y= i, palette="Reds")
    plt.show();


# In[36]:


print(df['target'].value_counts())
print('\n')
sns.countplot(x = df.target, palette=("Reds"));

