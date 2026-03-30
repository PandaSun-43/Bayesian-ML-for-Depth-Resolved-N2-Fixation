#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# In[2]:


dt = pd.read_csv("data_2.csv")
dt.columns


# In[46]:


dt['log_Fe'] = dt['Fe'].apply(lambda x: np.log10(x))
dt['log_N'] = dt['Nitrate'].apply(lambda x: np.log10(x))
dt['log_P'] = dt['Phosphate'].apply(lambda x: np.log10(x))
dt['log_CHL'] = dt['CHL(m)'].apply(lambda x: np.log10(x))


# In[37]:


dt['Total N2 Fixation (μmol N m-3 d-1)'].describe()
import matplotlib.pyplot as plt
plt.hist(dt['Total N2 Fixation (μmol N m-3 d-1)'], bins=20)


# In[41]:


dt.loc[dt['Total N2 Fixation (μmol N m-3 d-1)'] <= 0, 'Total N2 Fixation (μmol N m-3 d-1)'] = np.nan

dt['log_y'] = dt['Total N2 Fixation (μmol N m-3 d-1)'].apply(lambda x: np.log10(x))


# In[42]:


import matplotlib.pyplot as plt
plt.hist(dt['log_y'], bins=20)


# In[43]:


dt = dt.replace([np.inf, -np.inf], np.nan)
dt = dt.dropna(how='any')


# In[47]:


dt


# ## Temperature

# In[63]:


print(dt['DATE'].head())
print(dt['DATE'].dtype)


# In[64]:


import pandas as pd
import numpy as np
import glob

T_folder = r"/Users/huyiwen/Desktop/paper/T"
dt['T'] = np.nan

# 把 depth 列名提取出来（去掉前缀字符串）
def get_depth_columns(df):
    cols = df.columns
    depth_cols = []
    for c in cols:
        try:
            # 有的列名前面有空格，需要strip
            val = float(c.strip())
            depth_cols.append((c, val))
        except:
            continue
    return depth_cols

for month_num in range(1, 13): # woa23_decav_t01mn01
    csv_file = T_folder + "/woa23_decav_t" + str(month_num).rjust(2, '0') + "mn01.csv"
    try:
        dt_csv = pd.read_csv(csv_file, skiprows=1)
    except FileNotFoundError:
        continue

    # 提取深度列名及数值
    depth_cols = get_depth_columns(dt_csv)

    dt['DATE'] = pd.to_datetime(dt['DATE'], errors='coerce')

    idxes = dt.index[dt['DATE'].dt.month == month_num]
    if len(idxes) == 0:
        continue

    for idx in idxes:
        lat = dt.loc[idx, 'LATITUDE']
        lon = dt.loc[idx, 'LONGITUDE']
        depth = dt.loc[idx, 'DEPTH (m)']

        # 找最近的 lat/lon
        closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                     (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]

        # 找最近的 depth
        if len(depth_cols) > 0:
            col, col_depth = min(depth_cols, key=lambda x: abs(x[1] - depth))
            T_value = closest_point[col]
        else:
            T_value = np.nan

        dt.loc[idx, 'T'] = T_value


# In[65]:


dt


# In[88]:


dt = dt.replace([np.inf, -np.inf], np.nan)
dt = dt.dropna(how='any')


# In[89]:


dt


# ### MLD

# In[83]:


from scipy.io import loadmat
mld = loadmat('/Users/huyiwen/Desktop/stats304 project/Argo_mixedlayers_monthlyclim_04142022.mat')
print(mld.keys())


# In[84]:


print("latm:", mld['latm'].shape)
print("lonm:", mld['lonm'].shape)
print("mld_dt_mean:", mld['mld_dt_mean'].shape)


# In[86]:


from scipy.io import loadmat

mld = loadmat('/Users/huyiwen/Desktop/stats304 project/Argo_mixedlayers_monthlyclim_04142022.mat')
dt['MLD'] = np.nan

for mn_idx in range(0, 12):
    # Step 2: Match month with 'Date' column in 'dt' DataFrame
    dt['DATE'] = pd.to_datetime(dt['DATE'], errors='coerce')
    idxes = dt.index[dt['DATE'].dt.month == (mn_idx + 1)]

    if len(idxes) > 0:
        # Step 3: Find closest latitude and longitude values
        # print(dt_csv.columns)
        for idx in idxes:
            lat = dt.loc[idx, 'LATITUDE']
            lon = dt.loc[idx, 'LONGITUDE']
            lat_idx = np.abs(mld['latm'][0] - lat).argmin()
            lon_idx = np.abs(mld['lonm'][:,0] - lon).argmin()
            
            # Step 4: Add salinity value to 'dt' DataFrame
            dt.loc[idx, 'MLD'] = mld['mld_dt_mean'][mn_idx, lon_idx, lat_idx]


# In[87]:


dt


# In[ ]:





# In[90]:


env_vars = ['SST(m)', 'log_CHL', 'PAR(m)', 'salinity', 'log_N', 'log_P',
       'dissolved_oxygen', 'log_Fe', 'DEPTH (m)','T', 'MLD', 'coord1', 'coord2', 'coord3','log_y']

plt.figure(figsize=(15, 10))
for i, var in enumerate(env_vars, 1):
    plt.subplot(5, 3, i)
    plt.hist(dt[var].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    
    # Add log scale if needed (based on initial look)
    if dt[var].skew() > 1:
        plt.yscale('log')
        #plt.title(f'Distribution of {var} (log scale)')
        
plt.tight_layout()
#plt.savefig('env_vars.png')
plt.show()


# ## Group

# In[91]:


def determine_region(latitude, longitude):
    # 经度已在 [-180, 180] 范围内

    # NPac: 两部分
    if (30 <= latitude <= 85) and ((120 <= longitude <= 180) or (-180 <= longitude <= -90)):
        return "NPac"
    
    # NPacGyre:
    if (15 <= latitude <= 30) and ((100 <= longitude <= 180) or (-180 <= longitude <= -120)):
        return "NPacGyre"
    
    
    # EqPac
    if (0 <= latitude <= 15) and (100 <= longitude <= 180):
        return "EqPac"
    if (-20 <= latitude <= 15) and (-120 <= longitude <= -60):
        return "EqPac"
    if (-5 <= latitude <= 20) and (-180 <= longitude <= -120):  # 新增部分
        return "EqPac"
    if (0 <= latitude <= 40) and (-120 <= longitude <= -90):   # 新增部分
        return "EqPac"
    
    # SPacGyre: 第一部分
    if (-50 <= latitude <= 0) and (145 <= longitude <= 180):
        return "SPacGyre"
    # SPacGyre: 第二部分
    if (-50<= latitude <= -5) and (-180 <= longitude <= -100):
        return "SPacGyre"
    # SPacGyre: 第三部分
    if (-50 <= latitude <= -20) and (-100 <= longitude <= -60):
        return "SPacGyre"
    
    # NAtl
    if (40 <= latitude <= 85) and (-90 <= longitude <= 70):
        return "NAtl"
    
    # NAtlGyre
    if (15 <= latitude <= 40) and (-90 <= longitude <= 30):
        return "NAtlGyre"
    
    # EqAtl
    if (-10 <= latitude <= 15) and (-75 <= longitude <= 30):
        return "EqAtl"
    
    # SAtl
    if (-50 <= latitude <= -10) and (-60 <= longitude <= 30):
        return "SAtl"
    
    # NInd
    if (-10 <= latitude <= 40) and (30 <= longitude <= 145):
        return "NInd"
    
    # SInd
    if (-50 <= latitude <= -10) and (30 <= longitude <= 145):
        return "SInd"
    
    # SO
    if (-70 <= latitude <= -50):
        return "SO"
    
    return "unknown"


# In[92]:


dt['region'] = dt.apply(lambda row: determine_region(row['LATITUDE'], row['LONGITUDE']), axis=1)
region_counts = dt['region'].value_counts()
print(region_counts)


# In[93]:


# Define the region-to-group mapping
region_to_group = {
    'NPacGyre': 'group1',
    'NPac': 'group1',
    
    'EqAtl': 'group2',
    'SAtl': 'group2',
    
    'EqPac': 'group3',
    
    'SPacGyre': 'group4',
    
    'NAtl': 'group5',
    'NAtlGyre': 'group5',

    'NInd': 'group6',
    'SInd': 'group6',
    'SO': 'group6',


}

# Define the group-to-code mapping
group_to_code = {
    'group1': 0,
    'group2': 1,
    'group3': 2,
    'group4': 3,
    'group5': 4,
    'group6': 5,
    

}

# Apply the mappings to create 'group' and 'group_code' columns
dt['group'] = dt['region'].map(region_to_group)
dt['group_code'] = dt['group'].map(group_to_code)


# In[94]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# group
import seaborn as sns
unique_groups = dt['group'].unique()
palette = sns.color_palette("hls", len(unique_groups))
group_colors = dict(zip(unique_groups, palette))

# map
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

ax.coastlines()

ax.add_feature(cfeature.LAND, facecolor='lightgray')


for group in unique_groups:
    subset = dt[dt['group'] == group]
    ax.scatter(
        subset['LONGITUDE'], subset['LATITUDE'],
        color=group_colors[group],
        label=group,
        s=10,
        transform=ccrs.PlateCarree()
    )

plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Global Distribution of Points by Group')
plt.tight_layout()
plt.show()


# In[95]:


dt = dt.sort_values(by="group_code")


# In[96]:


dt["Total N2 Fixation (μmol N m-3 d-1)"] = pd.to_numeric(
    dt["Total N2 Fixation (μmol N m-3 d-1)"], errors="coerce"
)


# In[97]:


dt.to_csv("cleaned_data.csv", index=True)


# ### Training/Testing

# In[29]:


dt = pd.read_csv("cleaned_data.csv")
dt


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.hist(dt["Total N2 Fixation (μmol N m-3 d-1)"].dropna(), bins=50, color='teal', edgecolor='black')
plt.xlabel("Total N2 Fixation (μmol N m-3 d-1)")
plt.ylabel("Frequency")
plt.title("Histogram of Total N2 Fixation")
plt.show()


# In[31]:


col = "Total N2 Fixation (μmol N m-3 d-1)"

count_over_80 = (dt[col] > 30).sum()
total_count = dt[col].notna().sum()
ratio = count_over_80 / total_count * 100

print(f">80: {count_over_80}")
print(f"{ratio:.2f}%")


# In[38]:


dt_clean = dt[dt[col] <= 30].copy()
dt_clean.to_csv("cleaned_data_2.csv", index=False)


# In[39]:


dt_clean


# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_by_group(df, group_col, test_size, random_state=None):
    train_list = []
    test_list = []
    
    # Loop over each unique group
    for group in df[group_col].unique():
        # Filter rows belonging to the current group
        group_data = df[df[group_col] == group]
        
        # Split the data for the current group
        train_data, test_data = train_test_split(group_data, test_size=test_size, random_state=random_state)
        
        # Append to the train and test list
        train_list.append(train_data)
        test_list.append(test_data)
    
    # Concatenate all groups' train/test data
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    return train_df, test_df

# Usage
train_df, test_df = split_data_by_group(dt_clean, group_col='group_code', test_size=0.2, random_state=42)


# In[41]:


train_df


# In[42]:


train_df.to_csv("train_df.csv", index=True)
test_df.to_csv("test_df.csv", index=True)


# In[ ]:





# In[24]:


fixation_col = dt["Total N2 Fixation (μmol N m-3 d-1)"]
print("dtype:", fixation_col.dtype)

print("Unique sample values:", fixation_col.unique()[:20])

fixation_num = pd.to_numeric(fixation_col, errors="coerce")
print(fixation_num.describe())


# ## Plot

# In[21]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
import pandas as pd

# data
dt = pd.read_csv("cleaned_data_2.csv")

# group → biome
biome_to_group = {
    'NPacGyre': 'group1', 'NPac': 'group1',
    'EqAtl': 'group2', 'SAtl': 'group2',
    'EqPac': 'group3',
    'SPacGyre': 'group4',
    'NAtl': 'group5', 'NAtlGyre': 'group5',
    'NInd': 'group6', 'SInd': 'group6'
}

# group → biomes 
from collections import defaultdict
group_to_biomes = defaultdict(set)
for biome, group in biome_to_group.items():
    group_to_biomes[group].add(biome)

# group → marker 
group_markers = {
    'group1': 'o',   # circle
    'group2': 's',   # square
    'group3': '^',   # triangle
    'group4': 'D',   # diamond
    'group5': 'P',   # plus (filled)
    'group6': 'X'    # X mark
}


# In[23]:


dt


# In[25]:


from matplotlib.ticker import LogLocator
sns.set(style='white', context='notebook')


# In[29]:


# figure
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.set_global()

# group → short label
group_short = {
    'group1': 'G1',
    'group2': 'G2',
    'group3': 'G3',
    'group4': 'G4',
    'group5': 'G5',
    'group6': 'G6'
}

# marker
for group, marker in group_markers.items():
    subset = dt[dt["group"] == group]
    lat = subset["LATITUDE"].values
    lon = subset["LONGITUDE"].values
    fixation = subset["Total N2 Fixation (μmol N m-3 d-1)"].values

    sc = ax.scatter(
        lon, lat, c=fixation, cmap="viridis",
        norm=LogNorm(vmin=0.01, vmax=100),
        s=20, marker=marker, transform=ccrs.PlateCarree(),
        label=f"{group_short[group]} ({', '.join(sorted(group_to_biomes[group]))})"
    )

# colorbar
cbar = plt.colorbar(sc, orientation="vertical", fraction=0.03, pad=0.04, shrink=0.7)
cbar.set_label(r"Total N$_2$ Fixation ($\mu$mol N m$^{-3}$ d$^{-1}$)")



# ticks
xticks = [-180, -90, 0, 90, 180]
yticks = [-90, -45, 0, 45, 90]
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.set_xticklabels(['180°W', '90°W', '0°', '90°E', '180°E'])
ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'])

# grid lines
#gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# legend
ax.legend(loc='lower left', fontsize=7, title="Group (Biomes)", frameon=True)

plt.savefig('Spatial Distribution by Group.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# ## GEBCO_2023 Grid
# A global grid at 15 arc-second intervals. Originally published in April 2023. This is the fifth GEBCO grid developed through the Nippon Foundation-GEBCO Seabed 2030 Project.
# 
# The grid uses as a ‘base’ Version 2.5.5 of the SRTM15+ data set, augmented with the gridded bathymetric data sets developed by the four Seabed 2030 Regional Centers.
# 

# In[4]:


import xarray as xr

# 读取 GEBCO 数据
gebco = xr.open_dataset("gebco_2023/GEBCO_2023.nc")

print(gebco)


# In[15]:


import pandas as pd
import xarray as xr

# 2. data
dt = pd.read_csv("cleaned_data_2.csv")

lat_da = xr.DataArray(dt['LATITUDE'], dims="points")
lon_da = xr.DataArray(dt['LONGITUDE'], dims="points")

# 3) 从 GEBCO 插值对应深度 
depths = gebco["elevation"].sel(
                        lat=lat_da, lon=lon_da, 
                                method="nearest" # 最近邻查找，不是插值 → 不调用 scipy 
  
).values
# elevation --> Bathymetry
dt["Bathymetry"] = -depths

# 删除陆地（Bathymetry <= 0 表示不在海里）
dt = dt[dt["Bathymetry"] > 0]

# 删除浅水（<200m）
dt_case1 = dt[dt["Bathymetry"] >= 200].copy()

# 可选：去掉观测值为 0 的
# dt_case1 = dt_case1[dt_case1["Total N2 Fixation (μmol N m-3 d-1)"] > 0]

dt_case1.to_csv("cleaned_case1.csv", index=False)

print("Before:", len(dt))
print("After:", len(dt_case1))


# In[17]:


dt_case1


# In[41]:


fig, axes = plt.subplots(
    2, 1, figsize=(10, 12),  
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

titles = ["Before Filtering (All Data)", "After Filtering (Case I Only, Depth ≥ 200 m)"]

for ax, data, title in zip(axes, [before, after], titles):
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.set_global()
    ax.set_title(title, fontsize=13)

    # scatter by group
    for group, marker in group_markers.items():
        subset = data[data["group"] == group]
        if len(subset) == 0:
            continue

        lat = subset["LATITUDE"].values
        lon = subset["LONGITUDE"].values
        fixation = subset["Total N2 Fixation (μmol N m-3 d-1)"].values

        sc = ax.scatter(
            lon, lat, c=fixation, cmap="viridis",
            norm=LogNorm(vmin=0.01, vmax=100),
            s=18, marker=marker, transform=ccrs.PlateCarree(),
            label=group_short[group]
        )

    # ticks
    xticks = [-180, -90, 0, 90, 180]
    yticks = [-90, -45, 0, 45, 90]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(['180°W', '90°W', '0°', '90°E', '180°E'])
    ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'])

# shared colorbar
cbar = fig.colorbar(sc, ax=axes, orientation="vertical", shrink=0.75, pad=0.03)
cbar.set_label(r"Total N$_2$ Fixation ($\mu$mol N m$^{-3}$ d$^{-1}$)")

# legend
axes[0].legend(
    loc="lower left", fontsize=7,
    title="Groups", frameon=True
)

plt.savefig("Before_vs_After_CaseI_Filter_vertical.png", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# ### Dataset

# In[2]:


dt_case1 = pd.read_csv("cleaned_case1.csv")


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.hist(dt_case1["Total N2 Fixation (μmol N m-3 d-1)"].dropna(), bins=50, color='teal', edgecolor='black')
plt.xlabel("Total N2 Fixation (μmol N m-3 d-1)")
plt.ylabel("Frequency")
plt.title("Histogram of Total N2 Fixation")
plt.show()


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_by_group(df, group_col, test_size, random_state=None):
    train_list = []
    test_list = []
    
    # Loop over each unique group
    for group in df[group_col].unique():
        # Filter rows belonging to the current group
        group_data = df[df[group_col] == group]
        
        # Split the data for the current group
        train_data, test_data = train_test_split(group_data, test_size=test_size, random_state=random_state)
        
        # Append to the train and test list
        train_list.append(train_data)
        test_list.append(test_data)
    
    # Concatenate all groups' train/test data
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    return train_df, test_df

# Usage
train_df, test_df = split_data_by_group(dt_case1, group_col='group_code', test_size=0.2, random_state=42)


# In[5]:


train_df


# In[6]:


train_df.to_csv("train_df.csv", index=True)
test_df.to_csv("test_df.csv", index=True)


# In[ ]:




