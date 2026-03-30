#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
print("CUDA available?", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")


# In[ ]:


import pandas as pd
from tabpfn import TabPFNRegressor
#from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[ ]:


#1. read the data
train_df = pd.read_csv('train_df.csv')
test_df = pd.read_csv('test_df.csv')

#2. target cols
feature_cols =  ["MLD", "salinity", "dissolved_oxygen", "T", "SST(m)", 
                 "log_CHL", "PAR(m)", "log_N", "log_P", "log_Fe", 
                 "DEPTH (m)", "coord1", "coord2", "coord3", "time_sin", "time_cos"]
target_col = 'log_y'

#3. test and train sets
X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

X_test = test_df[feature_cols].values
y_test = test_df[target_col].values


# In[ ]:


#4. train, fit
reg = TabPFNRegressor(model_path = "/home/netID/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
                      device='auto')
reg.fit(X_train, y_train)

#5. predict
preds = reg.predict(X_test)

#6. evaluation metrics
print('Mean Squared Error (MSE):', mean_squared_error(y_test, preds))
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, preds))
print('R-squared (R^2):', r2_score(y_test, preds))


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, color='steelblue', alpha=0.6, label=f'R² = {r2_score(y_test, preds):.3f}')

max_val = max(np.max(y_test), np.max(preds))
min_val = min(np.min(y_test), np.min(preds))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

# labels
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted')
plt.legend()
plt.grid(True)

# Show 
plt.tight_layout()
#plt.savefig("observed_vs_predicted.png", dpi=300)
plt.show()


# In[ ]:


from tabpfn.model.loading import save_fitted_tabpfn_model

# save
save_fitted_tabpfn_model(reg, "my_reg.tabpfn_fit")


# In[ ]:


test_df['prediction'] = preds
test_df.to_csv('test_df_with_predictions.csv', index=False)  
print("Done: test_df_with_predictions.csv")


# In[ ]:




