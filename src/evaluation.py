#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import joblib


# In[ ]:


# ======================
# 1 data
# ======================

df = pd.read_csv("test_df_with_predictions.csv")

features = [
"MLD","salinity","dissolved_oxygen","T","SST(m)",
"log_CHL","PAR(m)","log_N","log_P","log_Fe",
"DEPTH (m)","coord1","coord2","coord3","time_sin","time_cos"
]

# ======================
# 2 scaling para
# ======================

scaling = pd.read_csv("feature_scaling.csv")
means = dict(zip(scaling["feature"], scaling["mean"]))
stds  = dict(zip(scaling["feature"], scaling["std"]))

# standardize
for f in features:
    df[f"{f}_z"] = (df[f] - means[f]) / stds[f]

predictors = [f"{f}_z" for f in features]

X = df[predictors].values


# ======================
# 3 LR prediction
# ======================

lr_model = joblib.load("lr_model.pkl")

df["LR_pred"] = lr_model.predict(X)


# ======================
# 4 RF prediction
# ======================

rf_model = joblib.load("rf_model.pkl")

df["RF_pred"] = rf_model.predict(X)


# ======================
# 5 BHLR prediction
# ======================

coef_df = pd.read_csv("bayesian_posterior_means_by_group.csv", index_col=0)

bhlr_preds = []

for _, row in df.iterrows():

    gnum = row["group"].replace("group","")
    col = f"G{gnum}"

    intercept = coef_df.loc["Intercept", col]
    betas = coef_df.loc[[f"beta{i}" for i in range(1,17)], col].values

    x = row[predictors].values

    pred = intercept + np.dot(betas, x)

    bhlr_preds.append(pred)

df["BHLR_pred"] = bhlr_preds

df.to_csv("test_df_with_model_predictions.csv", index=False)

print("Predictions completed.")


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="white", context="notebook")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "STIX", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13
})

# ==========================
# 1 Load data
# ==========================

df = pd.read_csv("test_df_with_model_predictions.csv")

y_true = df["log_y"].values

# ==========================
# 2 Models
# ==========================

models_to_compare = {
    "(a) LR": df["LR_pred"],
    "(b) RF": df["RF_pred"],
    "(c) BHLR": df["BHLR_pred"],
    "(d) TabPFN": df["prediction"]
}

# ==========================
# 3 Plot
# ==========================

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (name, preds) in zip(axes, models_to_compare.items()):

    y_pred = preds.values

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # scatter
    sns.scatterplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        color="steelblue",
        s=20
    )

    # 1:1 line
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        linewidth=1
    )

    ax.set_title(name)

    ax.set_xlabel(r"Log$_{10}$ Observed NF ($\mu$mol N m$^{-3}$ d$^{-1}$)")
    ax.set_ylabel(r"Log$_{10}$ Predicted NF ($\mu$mol N m$^{-3}$ d$^{-1}$)")

    # metrics box
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(y_true)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray"
        )
    )

plt.tight_layout()
plt.show()


# In[ ]:


df = pd.read_csv("test_df_with_model_predictions.csv")

df["obs"] = 10**df["log_y"]
df["LR"] = 10**df["LR_pred"]
df["RF"] = 10**df["RF_pred"]
df["BHLR"] = 10**df["BHLR_pred"]
df["TabPFN"] = 10**df["prediction"]

profile_data = df[df["DEPTH (m)"] <= 1000].copy()

# depth bins
depth_bins = [1,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500,750,1000]

mid_depths = []

profiles = {
    "Observed": [],
    "LR": [],
    "RF": [],
    "BHLR": [],
    "TabPFN": []
}

for i in range(len(depth_bins)-1):

    mask = (
        (profile_data["DEPTH (m)"] >= depth_bins[i]) &
        (profile_data["DEPTH (m)"] < depth_bins[i+1])
    )

    if mask.any():

        mid_depth = (depth_bins[i] + depth_bins[i+1]) / 2
        mid_depths.append(mid_depth)

        profiles["Observed"].append(profile_data.loc[mask,"obs"].median())
        profiles["LR"].append(profile_data.loc[mask,"LR"].median())
        profiles["RF"].append(profile_data.loc[mask,"RF"].median())
        profiles["BHLR"].append(profile_data.loc[mask,"BHLR"].median())
        profiles["TabPFN"].append(profile_data.loc[mask,"TabPFN"].median())


# -----------------------
# Plot
# -----------------------

fig, ax = plt.subplots(figsize=(8,9))  

cmap = "viridis"
norm = LogNorm(vmin=0.01, vmax=100)

# scatter background
sc = ax.scatter(
    profile_data["obs"],
    profile_data["DEPTH (m)"],
    c=profile_data["obs"],
    cmap=cmap,
    norm=norm,
    s=12,
    edgecolors="none"
)

# model profiles
ax.plot(profiles["Observed"], mid_depths,
        color="black", marker="o", lw=2, label="Observed", zorder=5)

ax.plot(profiles["LR"], mid_depths,
        color="#1f77b4",marker="s",  lw=1.5, label="LR")

ax.plot(profiles["RF"], mid_depths,
        color="#2ca02c", marker="^", lw=1.5, label="RF")

ax.plot(profiles["BHLR"], mid_depths,
        color="#ff7f0e", marker="D", lw=1.5, label="BHLR")

ax.plot(profiles["TabPFN"], mid_depths,
        color="#d62728", marker="o", lw=1.5, linestyle="--", label="TabPFN")

# axis settings
ax.set_xscale("log")
ax.set_xlim(0.01,100)

ax.set_ylim(0,1000)
ax.invert_yaxis()

ax.set_yscale("symlog", linthresh=10)
ax.set_yticks([1,10,100,1000])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel(r"N$_2$ Fixation ($\mu$mol N m$^{-3}$ d$^{-1}$)")
ax.set_ylabel("Depth (m)")

ax.grid(True, linestyle="--", alpha=0.4)

ax.legend()

# -----------------------
# colorbar
# -----------------------

cbar = fig.colorbar(
    sc,
    ax=ax,
    orientation="horizontal",
    pad=0.08,
    fraction=0.035,   
    shrink=0.7      
)

cbar.set_label(
    label=None,
    labelpad=8,
    fontsize=10
)

plt.tight_layout()

plt.show()


# In[ ]:


sns.set(style="white", context="notebook")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "STIX", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13
})

# ==========================
# 1 Load data
# ==========================
df = pd.read_csv("test_df_with_model_predictions.csv")
y_true = df["log_y"].values

# ==========================
# 2 Models
# ==========================
models_to_compare = {
    "(a) LR": df["LR_pred"],
    "(b) RF": df["RF_pred"],
    "(c) BHLR": df["BHLR_pred"],
    "(d) TabPFN": df["prediction"]
}

# ==========================
# 3 Uncertainty quantification
# ==========================

# std/credible interval
uncertainties = {}

# RF example: using per-tree predictions std
rf_model = joblib.load("rf_model.pkl")
X = df[[c for c in df.columns if "_z" in c]].values

rf_tree_preds = np.stack([t.predict(X) for t in rf_model.estimators_], axis=0)  # shape: (n_trees, n_samples)
rf_std = rf_tree_preds.std(axis=0)
uncertainties["(b) RF"] = rf_std

# BHLR example: using posterior samples if available
# posterior samples csv: shape (n_samples, n_data)
# posterior sample prediction
try:
    posterior_samples = pd.read_csv("bhlr_posterior_samples.csv")  # shape: n_samples x n_data
    bhlr_std = posterior_samples.values.std(axis=0)
    uncertainties["(c) BHLR"] = bhlr_std
except FileNotFoundError:
    uncertainties["(c) BHLR"] = np.full_like(df["BHLR_pred"], np.nan)

# LR / TabPFN: use NaN
uncertainties["(a) LR"] = np.full_like(df["LR_pred"], np.nan)
uncertainties["(d) TabPFN"] = np.full_like(df["prediction"], np.nan)

# ==========================
# 4 Plot predictions + uncertainty
# ==========================
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (name, preds) in zip(axes, models_to_compare.items()):
    y_pred = preds.values
    y_std = uncertainties.get(name, np.zeros_like(y_pred))  # default 0

    # scatter with errorbars (uncertainty)
    ax.errorbar(
        y_true,
        y_pred,
        yerr=y_std,
        fmt='o',
        markersize=4,
        ecolor='lightgray',
        elinewidth=1,
        capsize=2,
        color="steelblue"
    )

    # 1:1 line
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1)

    # metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    ax.set_title(name)
    ax.set_xlabel(r"Log$_{10}$ Observed NF ($\mu$mol N m$^{-3}$ d$^{-1}$)")
    ax.set_ylabel(r"Log$_{10}$ Predicted NF ($\mu$mol N m$^{-3}$ d$^{-1}$)")
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(y_true)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
    )

plt.tight_layout()
plt.show()

