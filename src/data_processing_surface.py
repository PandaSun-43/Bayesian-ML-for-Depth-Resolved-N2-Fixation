#!/usr/bin/env python
# coding: utf-8

# # Project data

# MODIS chl based:
# CHL (modis)
# SST (modis)
# PAR (modis)
# 
# SEAWIFS chl based:
# CHL (seawifs)
# SST (avhrr)
# PAR (seawifs)

# ### Decomposition

# In[ ]:


import tarfile
import gzip
import os

def extract_tar_with_hdf_gz(tar_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # tar 
    with tarfile.open(tar_file, 'r') as tar:
        for member in tar.getmembers():
            # hdf.gz
            if member.name.endswith('.hdf.gz'):
                with tar.extractfile(member) as f:
                    uncompressed_data = gzip.decompress(f.read())
                    hdf_file_name = os.path.splitext(os.path.basename(member.name))[0]
                    hdf_file_path = os.path.join(output_dir, hdf_file_name)
                    with open(hdf_file_path, 'wb') as hdf_file:
                        hdf_file.write(uncompressed_data)
                    print(f"已解压 {hdf_file_name}")


output_directory = "....path "

# tar
for i in range(2002,2024):
    tar_file_path = f"/Users/huyiwen/Desktop/stats304 project/CHL(modis)/chl.m.{i}.tar"
    extract_tar_with_hdf_gz(tar_file_path, output_directory)
print("ok")


# In[ ]:





# ## SST(m)

# In[7]:


from scipy.io import loadmat
import glob
import pandas as pd
import numpy as np


# In[8]:


# -------------- #
# 1. Constants
# -------------- #
sst_m_path = r"/Users/huyiwen/Desktop/stats304 project/SST(m)"

# -------------- #
# 2. Convert DATE from observations
# to 8-day unit used by satellite data.
# -------------- #
cols_keep = ["METHODS:   Sampling/Analysis", 
             "Incubation Time (hour)", 
             "DATE (yyyy-mm-dd)", 
             "LATITUDE", 
             "LONGITUDE", 
             "DEPTH (m)", 
             "Total N2 Fixation (μmol N m-3 d-1)"]

dt = pd.read_excel("/Users/huyiwen/Desktop/stats304 project/2023-Shao et al-DiazotrophsDatabase-20230626.xlsx", 
                   'N2_Fixation (Incub. 24 h)',
                   usecols=cols_keep)

dt["DATE_str"] = dt["DATE (yyyy-mm-dd)"].astype(str)

dt["DATE_str"] = dt["DATE_str"].str.replace(r"[-]\d+$", "", regex=True)

dt["DATE"] = pd.to_datetime(dt["DATE_str"], errors="coerce")

dt = dt.dropna(subset=["DATE"])

dt = dt.assign(DATE_str=lambda x: x['DATE'].astype(str).str.slice(0,10))\
       .assign(DATE=lambda x: pd.to_datetime(x['DATE_str']))\
       .assign(YRDY=lambda x: x['DATE'].dt.dayofyear - 1)\
       .assign(YRDY=lambda x: (x['YRDY']/8).astype('int') * 8 + 1)\
       .assign(YRSTR=lambda x: x['DATE'].dt.year.astype('str'),
               YRDY_STR=lambda x: x['YRDY'].astype('str').str.rjust(3,'0'),
               YRDY=lambda x: x['YRSTR'] + x['YRDY_STR']
              )\
       .drop(columns=['YRSTR', 'YRDY_STR'])

dt


# In[9]:


# -------------- #
# 3. To ease repeatly loading data,
# match data with the same date together
# -------------- #
for yrdy in dt['YRDY'].unique():
    # 3.1. load CHL
    #print(yrdy)
    sst_fname = glob.glob(sst_m_path + "/sst." + yrdy + ".hdf.mat")
    if len(sst_fname) == 0:
        continue
    #print(sst_fname[0])
   
    sst_m = loadmat(sst_fname[0])['dt']
    sst_m[sst_m<-3] = np.nan
    #print(sst.shape)
   
    # 3.2. Lat and Lon
    itv = 180 / sst_m.shape[0]
    sst_lat = np.linspace(90-itv/2, -90+itv/2, sst_m.shape[0])[np.newaxis,:]
    sst_lon = np.linspace(-180+itv/2, 180-itv/2, sst_m.shape[1])[np.newaxis,:]
   
    # 3.3. Match CHL
    idx = dt["YRDY"] == yrdy
    i = np.abs(dt.loc[idx,"LATITUDE"].to_numpy()[:,np.newaxis] - sst_lat).argmin(axis=1)
    j = np.abs(dt.loc[idx,"LONGITUDE"].to_numpy()[:,np.newaxis] - sst_lon).argmin(axis=1)
    dt.loc[idx,"SST(m)"] = sst_m[i, j]

import pandas as pd
dt = pd.read_excel("/Users/huyiwen/Desktop/stats304 project/2023-Shao et al-DiazotrophsDatabase-20230626.xlsx", 
                   'N2_Fixation_Integral (24 h)')
# In[10]:


dt


# In[73]:





# ## SST(a)

# In[11]:


sst_a_path = r"/Users/huyiwen/Desktop/stats304 project/SST(a)"

for yrdy in dt['YRDY'].unique():
    sst_a_fname = glob.glob(sst_a_path + "/sst." + yrdy + ".hdf.mat")
    if len(sst_a_fname) == 0:
        continue
    sst_a = loadmat(sst_a_fname[0])['dt']
    sst_a[sst_a < -3] = np.nan
    
    itv = 180 / sst_a.shape[0]
    sst_lat = np.linspace(90-itv/2, -90+itv/2, sst_a.shape[0])[np.newaxis,:]
    sst_lon = np.linspace(-180+itv/2, 180-itv/2, sst_a.shape[1])[np.newaxis,:]
   
    # 3.3. Match SST
    idx = dt["YRDY"] == yrdy
    i = np.abs(dt.loc[idx,"LATITUDE"].to_numpy()[:,np.newaxis] - sst_lat).argmin(axis=1)
    j = np.abs(dt.loc[idx,"LONGITUDE"].to_numpy()[:,np.newaxis] - sst_lon).argmin(axis=1)
    dt.loc[idx,"SST(a)"] = sst_a[i, j]


# In[12]:


sst_a_fname = sst_a_path + "/sst.1998001.hdf.mat"
sst_a = loadmat(sst_a_fname)['dt']
sst_a[sst_a < -3] = np.nan
import matplotlib.pyplot as plt
plt.imshow(sst_a)


# ## CHL(m) and CHL(s)

# In[13]:


chl_m_path = r"/Users/huyiwen/Desktop/stats304 project/CHL(m)"


# In[14]:


for yrdy in dt['YRDY'].unique():
    #CHL(m)
    chl_m_fname = glob.glob(chl_m_path + "/chl." + yrdy + ".hdf.mat")
    if len(chl_m_fname) == 0:
        continue
    chl_m = loadmat(chl_m_fname[0])['dt']
    chl_m[chl_m < -3] = np.nan
    
    itv = 180 / chl_m.shape[0]
    chl_lat = np.linspace(90-itv/2, -90+itv/2, chl_m.shape[0])[np.newaxis,:]
    chl_lon = np.linspace(-180+itv/2, 180-itv/2, chl_m.shape[1])[np.newaxis,:]
    
    idx = dt["YRDY"] == yrdy
    i = np.abs(dt.loc[idx, "LATITUDE"].to_numpy()[:, np.newaxis] - chl_lat).argmin(axis=1)
    j = np.abs(dt.loc[idx, "LONGITUDE"].to_numpy()[:, np.newaxis] - chl_lon).argmin(axis=1)
    dt.loc[idx, "CHL(m)"] = chl_m[i, j]


# In[15]:


chl_s_path = r"/Users/huyiwen/Desktop/stats304 project/CHL(s)"
for yrdy in dt['YRDY'].unique():  
    #CHL(s)
    chl_s_fname = glob.glob(chl_s_path + "/chl." + yrdy + ".hdf.mat")
    if len(chl_s_fname) == 0:
        continue
    chl_s = loadmat(chl_s_fname[0])['dt']
    chl_s[chl_s < -3] = np.nan
    
    itv = 180 / chl_m.shape[0]
    chl_lat = np.linspace(90-itv/2, -90+itv/2, chl_s.shape[0])[np.newaxis,:]
    chl_lon = np.linspace(-180+itv/2, 180-itv/2, chl_s.shape[1])[np.newaxis,:]
    
    idx = dt["YRDY"] == yrdy
    i = np.abs(dt.loc[idx, "LATITUDE"].to_numpy()[:, np.newaxis] - chl_lat).argmin(axis=1)
    j = np.abs(dt.loc[idx, "LONGITUDE"].to_numpy()[:, np.newaxis] - chl_lon).argmin(axis=1)
    dt.loc[idx, "CHL(s)"] = chl_s[i, j]


# ## PAR(m) and PAR(s)

# In[17]:


par_m_path = r"/Users/huyiwen/Desktop/stats304 project/PAR(m)"
par_s_path = r"/Users/huyiwen/Desktop/stats304 project/PAR(s)"


# In[18]:


for yrdy in dt['YRDY'].unique():
    #par(m)
    par_m_fname = glob.glob(par_m_path + "/par." + yrdy + ".hdf.mat")
    if len(par_m_fname) == 0:
        continue
    par_m = loadmat(par_m_fname[0])['dt']
    par_m[par_m < -3] = np.nan
    
    itv = 180 / par_m.shape[0]
    par_lat = np.linspace(90-itv/2, -90+itv/2, par_m.shape[0])[np.newaxis,:]
    par_lon = np.linspace(-180+itv/2, 180-itv/2, par_m.shape[1])[np.newaxis,:]
    
    idx = dt["YRDY"] == yrdy
    i = np.abs(dt.loc[idx, "LATITUDE"].to_numpy()[:, np.newaxis] - par_lat).argmin(axis=1)
    j = np.abs(dt.loc[idx, "LONGITUDE"].to_numpy()[:, np.newaxis] - par_lon).argmin(axis=1)
    dt.loc[idx, "PAR(m)"] = par_m[i, j]


# In[19]:


for yrdy in dt['YRDY'].unique():
    #par(m)
    par_s_fname = glob.glob(par_s_path + "/par." + yrdy + ".hdf.mat")
    if len(par_s_fname) == 0:
        continue
    par_s = loadmat(par_s_fname[0])['dt']
    par_s[par_s < -3] = np.nan
    
    itv = 180 / par_s.shape[0]
    par_lat = np.linspace(90-itv/2, -90+itv/2, par_s.shape[0])[np.newaxis,:]
    par_lon = np.linspace(-180+itv/2, 180-itv/2, par_s.shape[1])[np.newaxis,:]
    
    idx = dt["YRDY"] == yrdy
    i = np.abs(dt.loc[idx, "LATITUDE"].to_numpy()[:, np.newaxis] - par_lat).argmin(axis=1)
    j = np.abs(dt.loc[idx, "LONGITUDE"].to_numpy()[:, np.newaxis] - par_lon).argmin(axis=1)
    dt.loc[idx, "PAR(s)"] = par_s[i, j]


# In[20]:


dt

# Save DataFrame dt to an Excel file
dt.to_excel("/Users/huyiwen/Desktop/stats304 project/data_1.xlsx", index=False)
# In[23]:


dt['PAR(m)'].describe()


# In[25]:


dt['PAR(s)'].describe()


# ## Salinity

# In[29]:


salinity_folder = r"/Users/huyiwen/Desktop/stats304 project/salinity"
csv_file = salinity_folder + "/woa23_decav_s01mn01.csv"

# skiprows
dt_csv = pd.read_csv(csv_file, skiprows=1)

print(dt_csv.columns.tolist())


# In[32]:


import pandas as pd
import numpy as np
import glob

salinity_folder = r"/Users/huyiwen/Desktop/stats304 project/salinity"
dt['salinity'] = np.nan

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

for month_num in range(1, 13):
    csv_file = salinity_folder + "/woa23_decav_s" + str(month_num).rjust(2, '0') + "mn01.csv"
    try:
        dt_csv = pd.read_csv(csv_file, skiprows=1)
    except FileNotFoundError:
        continue

    # 提取深度列名及数值
    depth_cols = get_depth_columns(dt_csv)

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
            salinity_value = closest_point[col]
        else:
            salinity_value = np.nan

        dt.loc[idx, 'salinity'] = salinity_value


# In[33]:


dt

import pandas as pd
import numpy as np
import glob

# Step 1: Iterate over CSV files in the 'salinity' folder
salinity_folder = r"/Users/huyiwen/Desktop/stats304 project/salinity"
dt['salinity'] = np.nan

for month_num in range(1, 13):
    # Read CSV file
    csv_file = salinity_folder + "/woa23_decav_s" + str(month_num).rjust(2,'0') + "mn01.csv"
    if len(csv_file) == 0:
        continue   
    print(csv_file)
    
    dt_csv = pd.read_csv(csv_file, skiprows=1)
    print(dt_csv.shape)
    
    # Step 2: Match month with 'Date' column in 'dt' DataFrame
    # find the index
    idxes = dt.index[dt['DATE'].dt.month == month_num]

    if len(idxes) > 0:
        # Step 3: Find closest latitude and longitude values
        #print(dt_csv.columns)
        for idx in idxes:
            lat = dt.loc[idx, 'LATITUDE']
            lon = dt.loc[idx, 'LONGITUDE']
            closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                         (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]
            salinity_value = closest_point[' AND VALUES AT DEPTHS (M):0']

            # Step 4: Add salinity value to 'dt' DataFrame
            dt.loc[idx, 'salinity'] = salinity_value
            
# In[34]:


dt['salinity'].describe()


# In[ ]:





# ## Nitrate

# In[35]:


Nitrate_folder = r"/Users/huyiwen/Desktop/stats304 project/Nitrate"
dt['Nitrate'] = np.nan

for month_num in range(1, 13):
    # ✅ Step 2:
    csv_file = Nitrate_folder + "/woa23_all_n" + str(month_num).rjust(2, '0') + "mn01.csv"
    try:
        dt_csv = pd.read_csv(csv_file, skiprows=1)
    except FileNotFoundError:
        continue

    depth_cols = get_depth_columns(dt_csv)

    idxes = dt.index[dt['DATE'].dt.month == month_num]
    if len(idxes) == 0:
        continue
    print(csv_file)

    for idx in idxes:
        lat = dt.loc[idx, 'LATITUDE']
        lon = dt.loc[idx, 'LONGITUDE']
        depth = dt.loc[idx, 'DEPTH (m)']

        # Step 3: 找最近的 lat/lon
        closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                     (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]

        # Step 4: 找最近的 depth
        if len(depth_cols) > 0:
            col, col_depth = min(depth_cols, key=lambda x: abs(x[1] - depth))
            Nitrate_value = closest_point[col]
        else:
            Nitrate_value = np.nan

        # Step 5
        dt.loc[idx, 'Nitrate'] = Nitrate_value

# Step 1: Iterate over CSV files in the 'Nitrate' folder
Nitrate_folder = r"/Users/huyiwen/Desktop/stats304 project/Nitrate"
dt['Nitrate'] = np.nan

for month_num in range(1, 13):
    # Read CSV file
    csv_file = Nitrate_folder + "/woa23_all_n" + str(month_num).rjust(2,'0') + "mn01.csv"
    if len(csv_file) == 0:
        continue   
    print(csv_file)
    
    dt_csv = pd.read_csv(csv_file, skiprows=1)
    print(dt_csv.shape)
    
    # Step 2: Match month with 'Date' column in 'dt' DataFrame
    # find the index
    idxes = dt.index[dt['DATE'].dt.month == month_num]

    if len(idxes) > 0:
        # Step 3: Find closest latitude and longitude values
        #print(dt_csv.columns)
        for idx in idxes:
            lat = dt.loc[idx, 'LATITUDE']
            lon = dt.loc[idx, 'LONGITUDE']
            closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                         (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]
            Nitrate_value = closest_point[' AND VALUES AT DEPTHS (M):0']

            # Step 4: Add salinity value to 'dt' DataFrame
            dt.loc[idx, 'Nitrate'] = Nitrate_value
            
# In[36]:


dt['Nitrate'].describe()
import matplotlib.pyplot as plt
plt.hist(dt['Nitrate'], bins=20)


# In[37]:


dt


# ## P

# In[38]:


# Phosphate
import pandas as pd
import numpy as np

Phosphate_folder = r"/Users/huyiwen/Desktop/stats304 project/Phosphate"
dt['Phosphate'] = np.nan

# 提取深度列函数
def get_depth_columns(df):
    cols = df.columns
    depth_cols = []
    for c in cols:
        try:
            val = float(c.strip())   # 列名是 '5', '10', ...
            depth_cols.append((c, val))
        except:
            continue
    return depth_cols

for month_num in range(1, 13):
    csv_file = Phosphate_folder + "/woa23_all_p" + str(month_num).rjust(2, '0') + "mn01.csv"
    try:
        dt_csv = pd.read_csv(csv_file, skiprows=1)
    except FileNotFoundError:
        continue

    print(csv_file, dt_csv.shape)
    depth_cols = get_depth_columns(dt_csv)

    # 找到这个月的数据
    idxes = dt.index[dt['DATE'].dt.month == month_num]
    if len(idxes) == 0:
        continue

    for idx in idxes:
        lat = dt.loc[idx, 'LATITUDE']
        lon = dt.loc[idx, 'LONGITUDE']
        depth = dt.loc[idx, 'DEPTH (m)']

        # 最近的 lat/lon
        closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                     (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]

        # 最近的 depth
        if len(depth_cols) > 0:
            col, col_depth = min(depth_cols, key=lambda x: abs(x[1] - depth))
            Phosphate_value = closest_point[col]
        else:
            Phosphate_value = np.nan

        dt.loc[idx, 'Phosphate'] = Phosphate_value


# In[39]:


dt['Phosphate'].describe()
import matplotlib.pyplot as plt
plt.hist(dt['Phosphate'], bins=20)


# In[40]:


dt


# In[41]:


dt_csv.columns


# In[ ]:





# ## Oxygen

# In[ ]:


# 取 0-500m最小值, np.nanmin(dt, axis=1)

# Step 1: Iterate over CSV files in the 'dissolved_oxygen' folder
dissolved_oxygen_folder = r"/Users/huyiwen/Desktop/stats304 project/dissolved_oxygen"
dt['dissolved_oxygen'] = np.nan

for month_num in range(1, 13):
    
    # Read CSV file
    csv_file = dissolved_oxygen_folder + "/woa23_all_o" + str(month_num).rjust(2,'0') + "mn01.csv"
    if len(csv_file) == 0:
        continue
    
    print(csv_file)
    dt_csv = pd.read_csv(csv_file, skiprows=1)
    print(dt_csv.shape)
    
    # Step 2: Match month with 'Date' column in 'dt' DataFrame
    idxes = dt.index[dt['DATE'].dt.month == month_num]

    if len(idxes) > 0:
        # Step 3: Find closest latitude and longitude values
        # print(dt_csv.columns)
        for idx in idxes:
            lat = dt.loc[idx, 'LATITUDE']
            lon = dt.loc[idx, 'LONGITUDE']
            
            closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                         (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]
            
            relevant_columns = closest_point.loc[' AND VALUES AT DEPTHS (M):0':'500']
            
            dissolved_oxygen_value = np.nanmin(relevant_columns)

            # Step 4: Add salinity value to 'dt' DataFrame
            dt.loc[idx, 'dissolved_oxygen'] = dissolved_oxygen_value
# In[42]:


import pandas as pd
import numpy as np

dissolved_oxygen_folder = r"/Users/huyiwen/Desktop/stats304 project/dissolved_oxygen"
dt['dissolved_oxygen'] = np.nan

for month_num in range(1, 13):
    csv_file = dissolved_oxygen_folder + "/woa23_all_o" + str(month_num).rjust(2,'0') + "mn01.csv"
    try:
        dt_csv = pd.read_csv(csv_file, skiprows=1)
    except FileNotFoundError:
        continue
    
    print(csv_file, dt_csv.shape)
    depth_cols = get_depth_columns(dt_csv)
    
    # 找到这个月的数据
    idxes = dt.index[dt['DATE'].dt.month == month_num]
    if len(idxes) == 0:
        continue

    for idx in idxes:
        lat = dt.loc[idx, 'LATITUDE']
        lon = dt.loc[idx, 'LONGITUDE']
        depth = dt.loc[idx, 'DEPTH (m)']

        # 最近的 lat/lon
        closest_point = dt_csv.iloc[((dt_csv['#COMMA SEPARATED LATITUDE'] - lat)**2 + 
                                     (dt_csv[' LONGITUDE'] - lon)**2).idxmin()]
        
        # 最近的 depth
        if len(depth_cols) > 0:
            col, col_depth = min(depth_cols, key=lambda x: abs(x[1] - depth))
            dissolved_oxygen_value = closest_point[col]
        else:
            dissolved_oxygen_value = np.nan

        # 写入
        dt.loc[idx, 'dissolved_oxygen'] = dissolved_oxygen_value


# In[43]:


dt['dissolved_oxygen'].describe()
import matplotlib.pyplot as plt
plt.hist(dt['dissolved_oxygen'], bins=20)


# In[44]:


dt


# ### Fe

# In[51]:


# Step 1: Iterate over CSV files 
Fe_folder = r"/Users/huyiwen/Desktop/2024SRS/dFe_RF"
dt['Fe'] = np.nan  

# Fe 文件的深度层（和 CSV 列名一致）
Fe_depths = np.array([
    11.000000000000002, 25.000210762023926, 35.00164031982422, 45.005210876464844, 55.01295471191406, 
    65.02882766723634, 75.06052398681639, 85.12305450439452, 
    95.24561309814453, 105.48506164550784, 115.95201110839845, 126.8614730834961
])

for month_num in range(1, 13):
    csv_file = Fe_folder + "/dFe_RF_data_month_" + str(month_num).rjust(2,'0') + ".csv"
    if len(csv_file) == 0:
        continue  
    
    print(f"Processing {csv_file}")
    dt_csv = pd.read_csv(csv_file)
    
    # 找到该月的点
    idxes = dt.index[dt['DATE'].dt.month == month_num]

    if len(idxes) > 0:
        for idx in idxes:
            lat = dt.loc[idx, 'LATITUDE']
            lon = dt.loc[idx, 'LONGITUDE']
            depth_obs = dt.loc[idx, 'DEPTH (m)']  # 直接用采样深度
            
            # 找最近的格点
            closest_point = dt_csv.iloc[((dt_csv['latitude'] - lat)**2 + 
                                         (dt_csv['longitude'] - lon)**2).idxmin()]
            
            # 找最接近的 Fe 深度层
            closest_depth_idx = np.abs(Fe_depths - depth_obs).argmin()
            closest_depth = Fe_depths[closest_depth_idx]
            Fe_column = f"{closest_depth}"

            # 取 Fe 值
            Fe_value = closest_point.get(Fe_column, np.nan)
            dt.loc[idx, 'Fe'] = Fe_value


# In[52]:


dt


# ### Fe: N, N:P

# In[53]:


dt['Fe:N'] = dt['Fe'] / dt['Nitrate']
dt['N:P'] = dt['Nitrate'] / dt['Phosphate']


# ### Coordinates, time-sin/cos

# In[54]:


dt['coord1'] = np.sin(np.radians(dt['LATITUDE']))
dt['coord2'] = np.sin(np.radians(dt['LONGITUDE'])) * np.cos(np.radians(dt['LATITUDE']))
dt['coord3'] = -np.cos(np.radians(dt['LONGITUDE'])) * np.cos(np.radians(dt['LATITUDE']))

dt['month'] = pd.to_datetime(dt['DATE_str']).dt.month
dt['time_cos'] = np.cos(2 * np.pi * dt['month'] / 12)
dt['time_sin'] = np.sin(2 * np.pi * dt['month'] / 12)


# In[55]:


dt.columns


# In[56]:


select_cols = ['METHODS:   Sampling/Analysis', 'Incubation Time (hour)',
       'DATE (yyyy-mm-dd)', 'LATITUDE', 'LONGITUDE', 'DEPTH (m)',
       'Total N2 Fixation (μmol N m-3 d-1)', 'DATE_str', 'DATE', 'YRDY',
       'SST(m)', 'CHL(m)',  'PAR(m)', 'salinity',
       'Nitrate', 'Phosphate', 'dissolved_oxygen', 'Fe', 'Fe:N', 'N:P',
       'coord1', 'coord2', 'coord3', 'month', 'time_cos', 'time_sin']
df = dt[select_cols]


# In[58]:


# drop some rows which the N2 fixations are some strange strings
df.drop(df[df['Total N2 Fixation (μmol N m-3 d-1)'] == 'BLD'].index, inplace=True)
df.drop(df[df['Total N2 Fixation (μmol N m-3 d-1)'] == 'n.a'].index, inplace=True)
df.drop(df[df['Total N2 Fixation (μmol N m-3 d-1)'] == '< DL'].index, inplace=True)
df = df.dropna(how='any')


# In[60]:


df['Fe:N'] = df['Fe'] / df['Nitrate']
df['N:P'] = df['Nitrate'] / df['Phosphate']


# In[62]:


df


# In[ ]:




# Save DataFrame dt to an Excel file
dt.to_excel("/Users/huyiwen/Desktop/paper/data_1.xlsx", index=False)
# In[64]:


df.to_csv("/Users/huyiwen/Desktop/paper/data_2.csv", index=False)


# In[ ]:





# # Matching MLD

# In[45]:


from scipy.io import loadmat
mld = loadmat('/Users/huyiwen/Desktop/stats304 project/Argo_mixedlayers_monthlyclim_04142022.mat')
print(mld.keys())


# In[46]:


print("latm:", mld['latm'].shape)
print("lonm:", mld['lonm'].shape)
print("mld_dt_mean:", mld['mld_dt_mean'].shape)


# In[64]:


from scipy.io import loadmat

mld = loadmat('/Users/huyiwen/Desktop/stats304 project/Argo_mixedlayers_monthlyclim_04142022.mat')
dt['MLD'] = np.nan

for mn_idx in range(0, 12):
    # Step 2: Match month with 'Date' column in 'dt' DataFrame
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


# In[104]:


dt['MLD'].describe()


# In[65]:


dt


# In[66]:


# Save DataFrame dt to an Excel file
dt.to_excel("/Users/huyiwen/Desktop/stats304 project/data_2.xlsx", index=False)


# In[ ]:





# In[105]:


import netCDF4 as nc


# In[106]:


dt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




