#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
import arviz as az


# In[ ]:


warnings.filterwarnings("ignore", module="scipy")
RANDOM_SEED = 8924
az.style.use("arviz-darkgrid")


# In[ ]:


train_df = pd.read_csv("train_df.csv")
test_df = pd.read_csv("test_df.csv")


# In[ ]:


from sklearn.preprocessing import StandardScaler

features = ["MLD", "salinity", "dissolved_oxygen", "T", "SST(m)", "log_CHL", "PAR(m)", 
            "log_N", "log_P", "log_Fe", "DEPTH (m)", "coord1", "coord2", "coord3", "time_sin", "time_cos"]

scaler = StandardScaler()

# calculate mean/sd
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features]) 

# train_df , test_df
for i, feature in enumerate(features):
    train_df[f"{feature}_z"] = train_scaled[:, i]
    test_df[f"{feature}_z"] = test_scaled[:, i]

# mean/std 
scaling_params = pd.DataFrame({
    'feature': features,
    'mean': scaler.mean_,
    'std': scaler.scale_
})
scaling_params.to_csv("feature_scaling.csv", index=False)


# In[1]:


group, mn_groups = train_df.group.factorize()
coords = {"group": mn_groups}
coords


# In[ ]:


with pm.Model(coords=coords) as varying_intercept_slope:
    group_idx = pm.ConstantData("group_idx", train_df['group_code'].values, dims="obs_id")

    # Priors
    mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=2.0)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)  #gamma scale --> tau small ---> sigma variance bigger 
    # Priors for slopes
    mu_priors = [pm.Normal(f"mu{i+1}", mu=0.0, sigma=2.0) for i in range(16)]
    sigma_betas = [pm.HalfNormal(f"sigma_beta{i+1}", sigma=1.0) for i in range(16)] # mean of tau: small --> sigma2 large

    
    # Model error
    sigma_error = pm.HalfNormal("sigma_error", sigma=1.0)

    # --- HIERARCHICAL PARAMETERS ---
    # 1. offset（mean=0，var=1）
    alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, dims="group")
    betas_offset = [pm.Normal(f"beta_offset{i+1}", mu=0, sigma=1, dims="group") for i in range(16)]

    # 2. calculate alpha and betas
    alpha = pm.Deterministic("alpha", mu_alpha + alpha_offset * sigma_alpha, dims="group")
    betas = [pm.Deterministic(f"beta{i+1}", mu_priors[i] + betas_offset[i] * sigma_betas[i], dims="group") for i in range(16)]

    # Expected value
    features = ["MLD", "salinity", "dissolved_oxygen", "T", "SST(m)", "log_CHL", "PAR(m)", 
                "log_N", "log_P", "log_Fe", "DEPTH (m)", 
                "coord1", "coord2", "coord3", "time_sin", "time_cos"]
    
    y_hat = alpha[group_idx]
    for i, feature in enumerate(features):
        y_hat += betas[i][group_idx] * train_df[f"{feature}_z"].values

    # Data likelihood
    y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_error, observed=train_df['log_y'].values, dims="obs_id")
    


# In[ ]:


with varying_intercept_slope:
    trace1 = pm.sample(
    draws=2000,          # posterior
    tune=2000,           # burn-in
    chains=4,
    target_accept=0.95,   # hier model more stable
    random_seed=RANDOM_SEED
)


# In[ ]:


# Step 2: Get posterior means for prediction
post_mean1 = trace1.posterior.mean(dim=("chain", "draw"))
alpha_post_mean1 = post_mean1["alpha"].values
beta_post_means1 = [post_mean1[f"beta{i+1}"].values for i in range(16)]

# from posterior get para for each layer
mu_alpha = trace1.posterior["mu_alpha"].values.flatten()
sigma_alpha = trace1.posterior["sigma_alpha"].values.flatten()
mu_priors = [trace1.posterior[f"mu{i+1}"].values.flatten() for i in range(16)]
sigma_priors = [trace1.posterior[f"sigma_beta{i+1}"].values.flatten() for i in range(16)]

# sample alpha_new, beta_new
rng = np.random.default_rng()

# alpha_new ~ Normal(mu_alpha, sqrt(tau_alpha))
alpha_new = rng.normal(mu_alpha, np.sqrt(sigma_alpha))

beta_new = [
    rng.normal(mu_priors[i], np.sqrt(sigma_priors[i]))
    for i in range(16)
]


# In[ ]:


plt.figure(figsize=(10, 6))
plt.violinplot([alpha_new] + beta_new, showmeans=True)
plt.xticks(np.arange(1, 18), ["α"] + [f"β{i}" for i in range(1, 17)], rotation=45)
plt.ylabel("Coefficient value")
plt.title("Posterior samples for new group coefficients")
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np

# --- G1-G6 ---
n_groups = alpha_post_mean1.shape[0]  # 6
n_vars = 1 + len(beta_post_means1)    # 17
var_names = ["Intercept"] + [f"beta{i+1}" for i in range(len(beta_post_means1))]

data = np.zeros((n_vars, n_groups))
data[0, :] = alpha_post_mean1
for i in range(len(beta_post_means1)):
    data[i+1, :] = beta_post_means1[i]

# --- G7 ---
alpha_new_mean = np.mean(alpha_new)
beta_new_means = [np.mean(b) for b in beta_new]

# new col
new_col = np.array([alpha_new_mean] + beta_new_means)

# together
data_new = np.column_stack([data, new_col])
group_names = [f"G{i+1}" for i in range(n_groups)] + ["G7"]

df = pd.DataFrame(data_new, columns=group_names, index=var_names)

df.to_csv("bayesian_posterior_means_by_group.csv", float_format="%.6f")


# In[ ]:


groups = test_df['group_code'].unique()
#fig, ax = plt.subplots(nrows=1,ncols=1, figsize=[5,5])

test_df["prediction"] = 0

features_z = [f"{f}_z" for f in features]  # e.g. "SST_z", "MLD_z", etc.

for i, group_num in enumerate(groups):
    dataset = test_df[test_df['group_code'] == group_num].copy()
    
    dataset["prediction"] = alpha_post_mean1[i]
    for j, feature in enumerate(features_z):
        dataset["prediction"] += beta_post_means1[j][i] * dataset[feature].values
    
    test_df.loc[test_df['group_code'] == group_num, "prediction"] = dataset["prediction"]


# In[ ]:


# r2
r2 = r2_score(test_df["log_y"], test_df["prediction"])
print(f"R2 (Bayesian hierarchical): {r2:.3f}")


# In[ ]:


# Scatter plot
plt.figure(figsize=(5,5))
plt.scatter(test_df["log_y"], test_df["prediction"], alpha=0.5,label=f"Overall R2 = {r2:.3f}")
plt.plot([test_df["log_y"].min(), test_df["log_y"].max()],
         [test_df["log_y"].min(), test_df["log_y"].max()], 'r--')
plt.xlabel("True log N2 fixation")
plt.ylabel("Predicted log N2 fixation")
plt.title("Bayesian Hierarchical Model Prediction")
plt.grid(True)
plt.show()

