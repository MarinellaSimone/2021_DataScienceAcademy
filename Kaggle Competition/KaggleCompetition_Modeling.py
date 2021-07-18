#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import pandas as pd
import time
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from parameters import *
from sklearn.inspection import plot_partial_dependence
import pickle
import sys
import getopt
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder


# In[2]:


def current_milli_time():
    return round(time.time() * 1000)


# In[3]:


print('## Start parameter setup')
target_column_name = 'target'
filename_train_csv = r'C:\Users\marin\Downloads\train.csv (1).zip' 
filename_test_csv = r'C:\Users\marin\Downloads\test.csv.zip'
scoring_function_name = 'neg_log_loss'
filename_model = r'C:\Users\marin\Downloads\sample_submission.csv (1).zip'


inner_cv_folds = 5
outer_cv_folds = 4
is_sample_training = True
sampling_train_size = 5000
sampling_random_state = 42


# In[4]:


print('## Loading training dataset')
raw = pd.read_csv(filename_train_csv)

if (is_sample_training):
    raw, _ = train_test_split(
        raw,
        train_size = sampling_train_size,
        random_state = sampling_random_state,
        stratify = raw[target_column_name]
    )

target_classes = sorted(set(raw[target_column_name]))

print(raw[target_column_name].value_counts())
print(target_classes)


X = raw.drop([target_column_name], axis = 1)
y_ = raw[target_column_name]

y = LabelEncoder().fit_transform(y_)
oho = OneHotEncoder()
oho.fit_transform(y.reshape(-1,1)).toarray()


# In[5]:


pipe_dt_kmeans = Pipeline([
    ('smote', SMOTE()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('annotation', FeatureUnion(n_jobs = -1, transformer_list = [
        ('original', NoTransformer()),
        ('kmeans', ModelTransformer(KMeans()))
    ])),
    ('logshape', LogShape()),
    ('histgradientboostingclassifier', HistGradientBoostingClassifier())
])

param_grid_dt_kmeans = {
    "smote__k_neighbors": [2],
    "smote__kind": ['regular'], 
    "pca__n_components" :[0.99], 
    "pca__svd_solver": ['full'], 
    "annotation__kmeans__model__n_clusters": [5],
    "histgradientboostingclassifier__max_iter": [600],
    "histgradientboostingclassifier__validation_fraction": [None],
    "histgradientboostingclassifier__learning_rate": [0.01], 
    "histgradientboostingclassifier__max_depth": [24], 
    "histgradientboostingclassifier__min_samples_leaf": [24], 
    "histgradientboostingclassifier__max_leaf_nodes":[60],
    "histgradientboostingclassifier__random_state": [123],
    "histgradientboostingclassifier__verbose": [1]
}

#Pipe2
pipe_lr_kmeans = Pipeline([
    ('smote', SMOTE()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('annotation', FeatureUnion(n_jobs = -1, transformer_list = [
        ('original', NoTransformer()),
        ('kmeans', ModelTransformer(KMeans()))
    ])),
    ('logshape', LogShape()),
    ('logisticregression',LogisticRegression())
])

param_grid_lr_kmeans = {
    "smote__k_neighbors": [10],
    "pca__svd_solver": ['full'], 
    "annotation__kmeans__model__n_clusters": [5],
    "logisticregression__C": [0.0001, 0.001, 0.01, 0.1, 1, 10], 
    "logisticregression__random_state": [42]
}

#Pipe3
pipe_gb_kmeans = Pipeline([
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('annotation', FeatureUnion(n_jobs = -1, transformer_list = [
        ('original', NoTransformer()),
        ('kmeans', ModelTransformer(KMeans()))
    ])),
    ('logshape', LogShape()),
    ('gradientboostingclassifier', GradientBoostingClassifier())
])

param_grid_gb_kmeans = {
    "pca__n_components" :[0.99], 
    "pca__svd_solver": ['full'], 
    "annotation__kmeans__model__n_clusters": [5],
    "gradientboostingclassifier__learning_rate": [0.1, 0.2, 0.2], 
    "gradientboostingclassifier__min_samples_leaf": [50, 100, 150], 
    "gradientboostingclassifier__max_depth": [8, 10], 
    "gradientboostingclassifier__max_features": ['sqrt', 'auto'], 
}

#Pipe 4 
pipe_lg_kmeans = Pipeline([
    ('smote', SMOTE()),
    ('minmaxscaler', MinMaxScaler()),
    ('selectkbest', SelectKBest()),
    ('annotation', FeatureUnion(n_jobs = -1, transformer_list = [
        ('original', NoTransformer()),
        ('kmeans', ModelTransformer(KMeans()))
    ])),
    ('logshape', LogShape()),
    ('lgbmclassifier', LGBMClassifier())
])

param_grid_lg_kmeans = {
    "smote__k_neighbors": [10, 20],
    "selectkbest__k": [10, 20],
    "annotation__kmeans__model__n_clusters": [5],
    "lgbmclassifier__objective":['multiclass'],
    "lgbmclassifier__boosting_type": ['gbdt'],
    "lgbmclassifier__n_estimators": [2000],
    "lgbmclassifier__random_state": [2021],
    "lgbmclassifier__learning_rate": [3e-2],
    "lgbmclassifier__max_depth": [73],
    "lgbmclassifier__num_leaves": [42],
    "lgbmclassifier__subsample": [0.84327],
    "lgbmclassifier__colsample_bytree": [0.234],
    "lgbmclassifier__reg_alpha": [16.724382543126165], 
    "lgbmclassifier__reg_lambda": [4.4252351797809535],
    "lgbmclassifier__min_child_samples": [47],
    "lgbmclassifier__min_child_weight": [0.0004586402479388673],
    "lgbmclassifier__importance_type": ['gain']
}

##Pipe5
pipe_cat_kmeans = Pipeline([
    ('smote', SMOTE()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('annotation', FeatureUnion(n_jobs = -1, transformer_list = [
        ('original', NoTransformer()),
        ('kmeans', ModelTransformer(KMeans()))
    ])),
    ('logshape', LogShape()),
    ('catboostclassifier', CatBoostClassifier())
])

param_grid_cat_kmeans = {
    "smote__k_neighbors": [10],
    "pca__n_components" :[0.99], 
    "pca__svd_solver": ['full'], 
    "annotation__kmeans__model__n_clusters": [5],
    "catboostclassifier__iterations": [50],
    "catboostclassifier__random_state": [13],
    "catboostclassifier__loss_function":['MultiClass'],
    "catboostclassifier__learning_rate": [0.028396011768208232], 
    "catboostclassifier__reg_lambda": [53.36106387139166],
    "catboostclassifier__subsample": [0.21198881480700427],
    "catboostclassifier__random_strength": [32.17597787651307],
    "catboostclassifier__min_data_in_leaf": [11],
    "catboostclassifier__depth": [10]
}

##Pipe6 
pipe_xg_kmeans = Pipeline([
    ('smote', SMOTE()),
    ('minmaxscaler', MinMaxScaler()),
    ('pca', PCA()),
    ('annotation', FeatureUnion(n_jobs = -1, transformer_list = [
        ('original', NoTransformer()),
        ('kmeans', ModelTransformer(KMeans()))
    ])),
    ('logshape', LogShape()),
    ('xgbclassifier', XGBClassifier(use_label_encoder=False))
])

param_grid_xg_kmeans = {
    "smote__k_neighbors": [10],
    "pca__n_components" :[0.99], 
    "pca__svd_solver": ['full'], 
    "annotation__kmeans__model__n_clusters": [5],
    "xgbclassifier__learning_rate": [0.018, 0.02, 0.022], 
    "xgbclassifier__n_estimators": [800], 
     "xgbclassifier__eta": [0.05], 
    "xgbclassifier__max_depth": [5], 
    "xgbclassifier__subsample": [0.8], 
    "xgbclassifier__colsample_bytree":[0.8], 
    "xgbclassifier__reg_alpha": [0], 
    "xgbclassifier__min_child_weight": [1], 
    "xgbclassifier__eval_metric": ['mlogloss']
}

pipe = pipe_xg_kmeans
param_grid = param_grid_xg_kmeans


# In[6]:


# ====================== #

inner_cv = StratifiedKFold(n_splits = inner_cv_folds)
outer_cv = StratifiedKFold(n_splits = outer_cv_folds)

gscv = HalvingGridSearchCV(
    pipe,
    param_grid = param_grid,
    cv = inner_cv,
    scoring = scoring_function_name
)

print('## Hyperparameter search setup')
print(gscv)


print('## Running cross_val_score')
cvs = cross_val_score(
    gscv,
    X = X,
    y = y,
    scoring = scoring_function_name,
    cv = outer_cv
)


print(cvs)
print(cvs.mean())
print(cvs.std())


print('## Building the final model')
gscv.fit(X, y)


print('## Loading test dataset')
X_test = pd.read_csv(filename_test_csv)


print('## Predictions on test dataset')
pred = gscv.predict_proba(X_test)
pred_df = pd.DataFrame(
    data = pred,
    columns = target_classes)
pred_df.insert(0, 'id', X_test[['id']].values.ravel())


print(pred_df)


print('## Save predictions for Kaggle on CSV')
pred_df.to_csv(
    'kaggle_submission_' + str(current_milli_time()) + '.csv',
    index = False
)


print('## Best estimator')
print(gscv.best_estimator_)


print('## Best parameters')
print(gscv.best_params_)


# In[7]:


print(cvs)
print(cvs.mean())
print(cvs.std())


# In[ ]:




