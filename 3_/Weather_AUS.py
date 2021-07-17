#!/usr/bin/env python
# coding: utf-8

# **RAIN IN AUSTRALIA**

# *Importing the principle libraries*

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, plot_roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.tree import DecisionTreeClassifier


# *Importing the dataset from Kaggle and starting the first step of data preparation*

# In[2]:


df = pd.read_csv(r'C:\Users\marin\OneDrive\Desktop\weatherAUS.csv')
df.head(7)


# *Founding NA values in the prediction variable*

# In[4]:


df = df[df['RainTomorrow'].notna()]
df['RainTomorrow']

#notna permette di vedere se nel DataFrame ci sono valori nulli/mancanti 


# In[5]:


df.shape


# In[6]:


df['RainTomorrow'].value_counts()


# In[7]:


df['RainTomorrow'] = np.where(df['RainTomorrow'] == 'Yes', 1, 0)
df['RainTomorrow'].values.ravel()


# In[8]:


df2 = pd.get_dummies(df, columns= ['Location'])

#get dummies_ traforma la variabile location in un insieme di variabili continue

print(df2.head())


# *Founding missing values in the entire dataset and using the mean strategy*

# In[9]:


missing_data = df.isnull().sum(axis=0)
print(missing_data)


# In[11]:


numerics = ['int64', 'float64', 'int32']
df3 = df2.select_dtypes(include=numerics)
print(df3.columns)


# In[12]:


fill_NaN = SimpleImputer(missing_values= np.nan, strategy= 'mean')
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(df3))
imputed_DF.columns = df3.columns 
imputed_DF.index = df3.index
print(imputed_DF)


# In[13]:


df3.describe()


# In[14]:


imputed_DF.describe()


# In[15]:


imputed_DF.columns


# **MODELING**

# *Starting the modeling with 'imputed_DF', the cleaned dataset*

# In[17]:


X = imputed_DF.drop('RainTomorrow', axis=1)
X_scaled = StandardScaler().fit(X)
X_train = X_scaled.transform(X)

Y_train = imputed_DF[['RainTomorrow']]


# **First model_ Logistic Regression**

# In[19]:


model1 = LogisticRegression(max_iter= 500)
model1.fit(X_train, Y_train.values.ravel())
predictions = model1.predict(X_train)


# In[20]:


confusion1 = confusion_matrix(Y_train, predictions)
plot_confusion_matrix(model1, X_train, Y_train);
plot_roc_curve(model1, X_train, Y_train);
print("Logistic Regression")
print('\n')
accuracy1 = round(model1.score(X_train, Y_train) *100, 2)
print("The accuracy of LogisticRegression is {} %".format(accuracy1))
print('\n')
print(classification_report(Y_train, predictions))
print('\n')

fpr, tpr, thresholds  = roc_curve(Y_train, predictions)
area = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area))


# In[21]:


#cv_LogisticRegression

cv_model1 = cross_val_score(model1, X_train, Y_train.values.ravel(), cv=10)
print(cv_model1)


# In[18]:


p_grid = {
    "C": [1, 10, 100]
}

inner_cv = StratifiedKFold(n_splits= 5, shuffle=True, random_state= 40)
outer_cv = StratifiedKFold(n_splits = 4, shuffle= True, random_state= 41)

clf = GridSearchCV(estimator= model1, param_grid= p_grid, cv= inner_cv)
nested_score= cross_val_score(clf, X= X_train, y= Y_train.values.ravel(), cv = outer_cv)
score_model1 = nested_score.mean()
print("The performance score is {}".format(score_model1))
print('\n')

grid_model1 = clf.fit(X_train, Y_train.values.ravel())
print("The best estimator is {}".format(grid_model1.best_estimator_))
print('\n')
print("The best parameter is {}".format(grid_model1.best_params_))


# **Second model_ kNN**

# In[23]:


model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, Y_train.values.ravel())
predictions2 = model2.predict(X_train)


# In[20]:


confusion2 = confusion_matrix(Y_train, predictions2)
plot_confusion_matrix(model2, X_train, Y_train);
plot_roc_curve(model2, X_train, Y_train);

print("KNeighborsClassifier")
print('\n')
accuracy2 = round(model2.score(X_train, Y_train) *100, 2)
print("The accuracy of kNN is {} %".format(accuracy2))
print('\n')
print(classification_report(Y_train, predictions2))
print('\n')
print('The area under the roc curve is {}'.format(roc_auc_score(predictions2, Y_train)))


# In[21]:


cv_model2 = cross_val_score(model2, X_train, Y_train.values.ravel(), cv= 10) 
print(cv_model2) 


# In[22]:


p_grid = {
    "n_neighbors": [3, 5, 11, 19]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= 40)
outer_cv = StratifiedKFold(n_splits=4, shuffle= True, random_state= 41)

clf_knn = GridSearchCV(estimator= model2, param_grid=p_grid, cv= inner_cv)
nested_score_knn= cross_val_score(clf_knn, X= X_train, y=Y_train.values.ravel(), cv= outer_cv)

score_model2= nested_score_knn.mean()
print("The performance score is {}".format(score_model2))
print('\n')

grid_model2 = clf_knn.fit(X_train, Y_train.values.ravel())
print("The best estimator is {}".format(grid_model2.best_estimator_))
print('\n')
print("The best parameter is {}".format(grid_model2.best_params_))


# In[ ]:




