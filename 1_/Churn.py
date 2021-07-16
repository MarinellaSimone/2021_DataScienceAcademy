#!/usr/bin/env python
# coding: utf-8

# **CUSTOMER CHURN PREDICTION**

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

# In[3]:


df = pd.read_csv("train.csv")
df.head(7)


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# *Founding nulling values*

# In[9]:


dfr = df.isnull().sum(axis=0)
print(dfr)
print('\n')
print("There aren't null values")


# *Founding duplicates values*

# In[11]:


dup= df.duplicated(keep=False)
print(dup)
print('\n')
print("There aren't duplicates")


# **MODELING**

# In[12]:


X_= df[["account_length", "number_vmail_messages", "total_day_minutes", "total_day_calls", "total_day_charge", "total_eve_minutes", "total_eve_calls", "total_eve_charge", "total_night_minutes", "total_night_calls", "total_night_calls", "total_intl_minutes", "total_intl_calls", "total_intl_charge", "number_customer_service_calls"]]
Y_= df[["churn"]]
print(type(Y_))

#[] = serie, [[]] = dataframe


# In[14]:


Y__ = np.where(Y_ == 'yes', 1, 0)
Y_train = Y__.ravel()
Y_train

#to_numpy() modifica Y_train da DataFrame in Numpy.ndarrey 
#ravel modifica l'arrey in 1 dimensione
#np.where permette di sostituire le stringhe con valori numerici


# In[15]:


X_ss = StandardScaler().fit(X_)
X_train = X_ss.transform(X_)

#StandardScaler scala e trasforma la variabile X 


# In[16]:


type(X_train)


# In[17]:


X_train.shape


# In[18]:


type(Y_train)


# In[19]:


Y_train.shape


# In[20]:


df["churn"].value_counts()


# **First Model_ Logistic Regression**

# In[22]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

#fit allena il modello sui dati 


# In[23]:


accuracy_Logistic = round(logreg.score(X_train, Y_train) *100, 2)
print(accuracy_Logistic,"%")

#score() calcola la percentuale di affidabilità del modello


# In[25]:


X_predict = logreg.predict(X_train)
print(classification_report(Y_train, X_predict))


# In[31]:


print(confusion_matrix(Y_train, X_predict))
print('\n')
plot_confusion_matrix(logreg, X_train, Y_train);


# In[35]:


fpr, tpr, thresholds  = roc_curve(Y_train, X_predict)
print(auc(fpr, tpr))
print('\n')
plot_roc_curve(logreg, X_train, Y_train);


# In[36]:


recision, recall, thresholds = precision_recall_curve(Y_train, X_predict)
plot_precision_recall_curve(logreg, X_train, Y_train);


# *Cross validation_ standard/stratified*

# In[37]:


cvs_logreg = cross_val_score(logreg, X_train, Y_train, cv=10)
print(cvs_logreg)


# In[38]:


skf = StratifiedKFold(n_splits=10)
scores = []

for train_index, test_index in skf.split(X_train, Y_train):
    X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
    Y_train_cv, Y_test_cv = Y_train[train_index], Y_train[test_index]
    
    logreg= LogisticRegression(max_iter= 5000)
    logreg.fit(X_train_cv, Y_train_cv)
    scores.append(logreg.score(X_test_cv, Y_test_cv))

scores


# *GridSearch with StratifiedkFold_ Logistic regression*

# In[39]:


p_grid = {
    "C": [1, 10, 100]
}

inner_cv = StratifiedKFold(n_splits= 5, shuffle=True, random_state= 40)
outer_cv = StratifiedKFold(n_splits = 4, shuffle= True, random_state= 41)

logreg = LogisticRegression()
clf = GridSearchCV(estimator= logreg, param_grid= p_grid, cv= inner_cv)
nested_score= cross_val_score(clf, X= X_train, y= Y_train, cv = outer_cv)
nested_score.mean()


# In[41]:


cross_validate_results= cross_validate(clf, X= X_train, y= Y_train, cv = outer_cv, scoring= ["accuracy", "precision"])
cross_validate_results


# In[42]:


gscv = clf.fit(X_train, Y_train)
print(gscv.best_estimator_)
print('\n')
print(gscv.best_params_)
print('\n')
print(gscv.cv_results_)


# **Second Model_ kNN**

# In[44]:


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)


# In[45]:


accuracy_Neigh = round(neigh.score(X_train, Y_train) *100, 2)
print(accuracy_Neigh, "%")


# In[46]:


X_predictions= neigh.predict(X_train)
print(classification_report(Y_train, X_predictions));


# In[48]:


print(confusion_matrix(Y_train, X_predictions))
print('\n')
plot_confusion_matrix(neigh, X_train, Y_train);


# In[51]:


fpr, tpr, thresholds = roc_curve(Y_train, X_predictions)
print(auc(fpr, tpr))
print('\n')
plot_roc_curve(neigh, X_train, Y_train);


# In[52]:


recision, recall, thresholds = precision_recall_curve(Y_train, X_predictions)
plot_precision_recall_curve(neigh, X_train, Y_train);


# *Cross validation_ standard*

# In[53]:


cvs_knn= cross_val_score(neigh, X_train, Y_train, cv= 10)
cvs_knn


# *Comparison of the two models in terms of accuracy*

# In[56]:


accuracy_models= [accuracy_Logistic, accuracy_Neigh]
print(accuracy_models)


# *Gridsearch with StratifiedkFold_ kNN*

# In[58]:


p_grid = {
    "n_neighbors": [3, 5, 11, 19]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= 40)
outer_cv = StratifiedKFold(n_splits=4, shuffle= True, random_state= 41)

neigh = KNeighborsClassifier()
clf_knn = GridSearchCV(estimator= neigh, param_grid=p_grid, cv= inner_cv)
nested_score_knn= cross_val_score(clf_knn, X= X_train, y=Y_train, cv= outer_cv)

nested_score_knn.mean()


# In[59]:


gs_results= clf_knn.fit(X_train, Y_train)
print(gs_results.best_score_)
print("\n")
print(gs_results.best_params_)
print("\n")
print(gs_results.cv_results_)

