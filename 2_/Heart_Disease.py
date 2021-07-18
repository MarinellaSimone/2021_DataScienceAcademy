#!/usr/bin/env python
# coding: utf-8

# **HEART DISEASE UCI**

# *Importing the principle libraries* 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import roc_curve, plot_roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC 
from xgboost import plot_importance
from sklearn.feature_selection import RFE, RFECV
from sklearn import tree
from sklearn import metrics


# *Importing the dataset from Kaggle and starting the first step of data preparation*

# In[2]:


df = pd.read_csv(r'C:\Users\marin\OneDrive\Desktop\heart.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.rename(columns ={'age':'Age','sex':'Sex','cp':'Chest_pain','trestbps':'Resting_blood_pressure','chol':'Cholesterol','fbs':'Fasting_blood_sugar',
                    'restecg':'ECG_results','thalach':'Maximum_heart_rate','exang':'Exercise_induced_angina','oldpeak':'ST_depression','ca':'Major_vessels',
                   'thal':'Thalassemia_types','target':'Heart_attack','slope':'ST_slope'}, inplace = True)


# In[5]:


print(df['Heart_attack'].value_counts())
print('\n')
sns.countplot(x = df.Heart_attack, palette=("crest_r"));


# *Founding missing values*

# In[6]:


missing_data = np.isnan(df).sum(axis=0)
print(missing_data)
print('\n')
print(missing_data.value_counts())
print('\n')
print("There aren't missing values in the dataset")


# *Founding duplicated values and drop them*

# In[7]:


duplicated_data = df.duplicated(keep=False)
print(duplicated_data)
print('\n')
print(duplicated_data.value_counts())
print('\n')
print("There are two duplicates")


# In[8]:


df[duplicated_data]


# In[9]:


df.drop_duplicates()


# In[10]:


def get_df_numerical(df, numerics = ['int64', 'float64']):
    return df.select_dtypes(include=numerics)

df2 = get_df_numerical(df)


# In[11]:


df3 = df2.drop("Heart_attack", axis =1)
df3_ = StandardScaler().fit(df3)
df3_scaled = df3_.transform(df3)


# *Founding the outlier values with different methods*

# *Outlier*

# In[12]:


for i in ["Age", "Resting_blood_pressure", "Cholesterol", "Maximum_heart_rate", "ST_depression"]:
    print(i.upper())
    sns.boxplot(data =df3, y= i, palette="crest_r")
    plt.show();


# *ScatterPlot*

# In[13]:


for i in ["Age", "Resting_blood_pressure", "Cholesterol", "Maximum_heart_rate", "ST_depression"]:
    print(i.upper())
    x = df2[i]
    
    plt.subplots(figsize=(8,4))
    plt.scatter(x, df2["Heart_attack"]) 
    plt.xlabel('Age')
    plt.ylabel('')
    plt.show()


# *Z-Score*

# In[14]:


for i in ["Age", "Resting_blood_pressure", "Cholesterol", "Maximum_heart_rate", "ST_depression"]:
    print(i.upper())
    print('\n')
    z = np.abs(stats.zscore(df2[i]))
    print(z)
    print('\n')
    print(np.where(z > 3))
    print('\n')


# *IQR*

# In[15]:


for i in ["Age", "Resting_blood_pressure", "Cholesterol", "Maximum_heart_rate", "ST_depression"]:
    print(i.upper())
    
    Q1 = np.percentile(df2[i], 25,
                       interpolation = 'midpoint') 
    
    Q3 = np.percentile(df2[i], 75,
                       interpolation = 'midpoint') 
    
    IQR = Q3 - Q1 
    
    upper = df2[i] >= (Q3+1.5*IQR)
    #print("Upper bound:",upper)
    print('\n')
    print(np.where(upper))
    print('\n')
    
    lower = df2[i] <= (Q1-1.5*IQR)
    #print("Lower bound:", lower)
    print(np.where(lower))
    print('\n')


# *On my opinion the outlier that could be eliminated are [28] in Cholesterol and [221] in St_Depression.*
# *Howevere, in this case, removing them isn't very significant*

# **MODELING**

# In[16]:


X_train = df3_scaled
Y_train = df2[["Heart_attack"]]


# *Splitting data into training and testing set*

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state=30)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# **First model_ Logistic Regression**

# In[18]:


model1 = LogisticRegression()
model1.fit(X_train, Y_train.values.ravel())
predictions = model1.predict(X_test)


# In[19]:


confusion1 = confusion_matrix(Y_test, predictions)
plot_confusion_matrix(model1, X_test, Y_test);
roc1 = plot_roc_curve(model1, X_test, Y_test);

print("Logistic Regression".upper())
print('\n')
accuracy1 = round(model1.score(X_test, Y_test) *100, 2)
print("The accuracy of Logistic Regression is {} %".format(accuracy1))
print('\n')
print(classification_report(Y_test, predictions))
print('\n')

fpr, tpr, thresholds  = roc_curve(Y_test, predictions)
area = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area))


# *cv_ Logistic Regression*

# In[20]:


cv_model1 = cross_val_score(model1, X_train, Y_train.values.ravel(), cv=10)
print("Cross Validation_Logistic Regression".upper())
print('\n')
print(cv_model1)

one = np.mean(cv_model1)
print('\n')
print("The mean is {}".format(one*100))


# **Second model_ kNN**

# In[21]:


model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, Y_train.values.ravel())
predictions2 = model2.predict(X_test)


# In[22]:


confusion2 = confusion_matrix(Y_test, predictions2)
plot_confusion_matrix(model2, X_test, Y_test);
roc2 = plot_roc_curve(model2, X_test, Y_test);

print("KNeighbors Classifier".upper())
print('\n')
accuracy2 = round(model2.score(X_test, Y_test) *100, 2)
print("The accuracy of kNN is {} %".format(accuracy2))
print('\n')
print(classification_report(Y_test, predictions2))
print('\n')

fpr, tpr, thresholds  = roc_curve(Y_test, predictions2)
area2 = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area2))


# *cv_ kNN*

# In[23]:


cv_model2 = cross_val_score(model2, X_train, Y_train.values.ravel(), cv= 10) 
print("Cross Validation_kNN".upper())
print('\n')
print(cv_model2) 

two = np.mean(cv_model2)
print('\n')
print("The mean is {}".format(two*100))


# **Third model_ Decision Tree Classifier**

# In[24]:


model3 = DecisionTreeClassifier(criterion = 'gini', random_state=0)
d_tree = model3.fit(X_train, Y_train)
predictions3 = model3.predict(X_test)

tree.plot_tree(d_tree);


# In[25]:


confusion3 = confusion_matrix(Y_test, predictions3)
plot_confusion_matrix(model3, X_test, Y_test);
roc3 = plot_roc_curve(model3, X_test, Y_test);

print("Decision Tree Classifier".upper())
print('\n')
accuracy3 = round(model3.score(X_test, Y_test) *100, 2)
print("The accuracy of DecisionTree is {} %".format(accuracy3))
print('\n')
print(classification_report(Y_test, predictions3))
print('\n')

fpr, tpr, thresholds  = roc_curve(Y_test, predictions3)
area3 = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area3))


# *cv_DecisionTree*

# In[26]:


cv_model3 = cross_val_score(model3, X_train, Y_train.values.ravel(), cv= 10) 
print("Cross Validation_Decision Tree Classifier".upper())
print('\n')
print(cv_model3) 

three = np.mean(cv_model3)
print('\n')
print("The mean is {}".format(three*100))


# **Fourth model_ Random Forest Classifier**

# In[27]:


model4 = RandomForestClassifier (n_estimators=100)
model4.fit(X_train, Y_train.values.ravel())
predictions4 = model4.predict(X_test)


# In[28]:


confusion4 = confusion_matrix(Y_test, predictions4)
plot_confusion_matrix(model4, X_test, Y_test);
roc4 = plot_roc_curve(model4, X_test, Y_test);

print("Random Forest Classifier".upper())
print('\n')
accuracy4 = round(model4.score(X_test, Y_test) *100, 2)
print("The accuracy of Random Forest is {} %".format(accuracy4))
print('\n')
print(classification_report(Y_test, predictions4))
print('\n')

curve_4 = fpr, tpr, thresholds  = roc_curve(Y_test, predictions4)
area4 = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area4))


# *cv_RandomForestClassifier*

# In[29]:


cv_model4 = cross_val_score(model4, X_train, Y_train.values.ravel(), cv= 10) 
print("Cross Validation_Random Forest Classifier".upper())
print('\n')
print(cv_model4) 

four = np.mean(cv_model4)
print('\n')
print("The mean is {}".format(four*100))


# **Fifth model_ AdaBoost**

# In[30]:


model5 = AdaBoostClassifier(n_estimators=100, random_state=0)
model5.fit(X_train, Y_train.values.ravel())
predictions5= model5.predict(X_test)


# In[31]:


confusion5 = confusion_matrix(Y_test, predictions5)
plot_confusion_matrix(model5, X_test, Y_test);
roc5 = plot_roc_curve(model5, X_test, Y_test);

print("Boosting".upper())
print('\n')
accuracy5 = round(model5.score(X_test, Y_test) *100, 2)
print("The accuracy of Boosting is {} %".format(accuracy5))
print('\n')
print(classification_report(Y_test, predictions5))
print('\n')

fpr, tpr, thresholds  = roc_curve(Y_test, predictions5)
area5 = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area5))


# *cv_ AdaBoost*

# In[32]:


cv_model5 = cross_val_score(model5, X_train, Y_train.values.ravel(), cv= 10) 
print("Cross Validation_Boosting".upper())
print('\n')
print(cv_model5) 

five = np.mean(cv_model5)
print('\n')
print("The mean is {}".format(five*100))


# **Sixth model_ Gradient Boosting**

# In[33]:


model6= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model6.fit(X_train, Y_train.values.ravel())
predictions6= model6.predict(X_test)


# In[34]:


confusion6 = confusion_matrix(Y_test, predictions6)
plot_confusion_matrix(model6, X_test, Y_test);
roc6 = plot_roc_curve(model6, X_test, Y_test);

print("Gradient Boosting".upper())
print('\n')
accuracy6 = round(model6.score(X_test, Y_test) *100, 2)
print("The accuracy of Gradient Boosting is {} %".format(accuracy6))
print('\n')
print(classification_report(Y_test, predictions6))
print('\n')

fpr, tpr, thresholds  = roc_curve(Y_test, predictions6)
area6 = auc(fpr, tpr)
print("The area under the roc curve is {}".format(area6))


# *cv_ GradientBoostingClassifier*

# In[35]:


cv_model6 = cross_val_score(model6, X_train, Y_train.values.ravel(), cv= 10) 
print("Cross Validation_Gradient Boosting".upper())
print('\n')
print(cv_model6) 

six = np.mean(cv_model6)
print('\n')
print("The mean is {}".format(six*100))


# **Seventh model_ XGBoost**

# In[36]:


model7= xgb.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
model7.fit(X_train, Y_train.values.ravel())
predictions7= model7.predict(X_test)


# In[37]:


mean_squared_error(Y_test, predictions7)


# **THE BEST MODEL**

# In[38]:


model_list = ["LogisticRegression", "kNN", "DecisionTree", "RandomForest", "Boosting", "GradientBoosting"]
accuracy_list = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6]

accuracy_models = pd.Series(data= accuracy_list, index=model_list)
print("Accuracy".upper())
print('\n')
print(accuracy_models)
print('\n')

plt.figure(figsize=(11,6));
sns.set_style('dark');
accuracy_plot= sns.barplot(x= model_list, y= accuracy_list, palette= ("BrBG_r"))
plt.title("Accuracy score among different models".upper());


# In[39]:


cv_list = [one*100, two*100, three*100, four*100, five*100, six*100]

cv_models= pd.Series(data=cv_list, index= model_list)
print("Cross Validation".upper())
print('\n')
print(cv_models)
print('\n')

plt.figure(figsize=(11,6));
sns.set_style('dark');

accuracy_plot= sns.barplot(x= model_list, y= cv_list, palette= ("BrBG_r"))
plt.title("Cross Validation among different models".upper());


# *The best model according to accuracy is the first model of Logistic Regression.*
# *The best model according to cv results is the second model of kNN.*
# *About this two model we can do a GridSearch and we can found the best parameter and score of each one.*

# In[43]:


p_grid = {
    "C": [1, 10, 14, 19, 100]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= 40)
outer_cv = StratifiedKFold(n_splits=4, shuffle= True, random_state= 41)

clf_logreg = GridSearchCV(estimator= model1, param_grid=p_grid, cv= inner_cv)
nested_score_logreg = cross_val_score(clf_logreg, X= X_test, y=Y_test.values.ravel(), cv= outer_cv)

score_model1 = nested_score_logreg.mean()
print("The performance score is {}".format(score_model1))
print('\n')

grid_model1 = clf_logreg.fit(X_test, Y_test.values.ravel())
print("The best score is {}".format(grid_model1.best_score_))
print('\n')
print("The best parameter is {}".format(grid_model1.best_params_))


# In[44]:


p_grid = {
    "n_neighbors": [3, 5, 11, 19, 26]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= 40)
outer_cv = StratifiedKFold(n_splits=4, shuffle= True, random_state= 41)

clf_knn = GridSearchCV(estimator= model2, param_grid=p_grid, cv= inner_cv)
nested_score_knn= cross_val_score(clf_knn, X= X_test, y=Y_test.values.ravel(), cv= outer_cv)

score_model2= nested_score_knn.mean()
print("The performance score is {}".format(score_model2))
print('\n')

grid_model2 = clf_knn.fit(X_test, Y_test.values.ravel())
print("The best score is {}".format(grid_model2.best_score_))
print('\n')
print("The best parameter is {}".format(grid_model2.best_params_))


# In[ ]:




