#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import arviz as az


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


# In[ ]:


## linear regression


# In[ ]:


features = ["MLD", "salinity", "dissolved_oxygen", "T", "SST(m)", "log_CHL", "PAR(m)", 
                "log_N", "log_P", "log_Fe", "DEPTH (m)", 
                "coord1", "coord2", "coord3", "time_sin", "time_cos"]
predictors_1 = [f"{f}_z" for f in features]
target = 'log_y'

# Split the data into training and test sets
train_x1 = train_df[predictors_1]
train_y1 = train_df[target]
test_x1 = test_df[predictors_1]
test_y1 = test_df[target]

# Linear Regression Model
mod_lr = LinearRegression()
mod_lr.fit(train_x1, train_y1)
y_pred_lr1 = mod_lr.predict(test_x1)
r2_score(test_y1, y_pred_lr1)


# In[ ]:


coefficients = mod_lr.coef_
intercept = mod_lr.intercept_

coef_df = pd.DataFrame({
    'Feature': train_x1.columns,
    'Coefficient': coefficients
})
intercept_df = pd.DataFrame({
    'Feature': ['Intercept'],
    'Coefficient': [intercept]
})
coef_df = pd.concat([intercept_df, coef_df], ignore_index=True)
coef_df


# In[ ]:


## Random Forests


# In[ ]:


mod_rf = RandomForestRegressor(n_estimators=100, 
                               max_depth=10,
                               min_samples_split=0.05,
                               max_features=0.5,
                               random_state=42)
mod_rf.fit(X=train_x1, y=train_y1)

test_y_pred1 = mod_rf.predict(test_x1)
r2_score(test_y1, test_y_pred1)


# In[ ]:


importances = mod_rf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': train_x1.columns,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df


# In[ ]:


import joblib
joblib.dump(mod_lr, "lr_model.pkl")
joblib.dump(mod_rf, "rf_model.pkl")


# In[ ]:




