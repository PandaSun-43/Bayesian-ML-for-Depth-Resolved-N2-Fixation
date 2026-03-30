# Data Dictionary

This document describes the variables used in the marine N₂ fixation prediction dataset.

## 📌 Target Variable

* **log_y**: Log-transformed total nitrogen fixation rate.

  * Original unit: μmol N m⁻³ d⁻¹
  * Transformation: log10
  * Description: Represents biological nitrogen fixation in the ocean, a key process in the global nitrogen cycle.

---

## 📌 Feature Variables

### 🌊 Physical and Biogeochemical Variables

* **MLD** : Mixed Layer Depth (m). Indicates upper ocean stratification. [https://mixedlayer.ucsd.edu/]

* **DEPTH (m)**: Sampling depth in meters.

* **T** : In situ temperature (°C) at depth. [https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23.pl?parameterOption=t]

* **SST(m)** : Sea Surface Temperature (°C). [https://www.ncei.noaa.gov/products/avhrr-pathfinder-sst]

* **salinity**: Seawater salinity (PSU). [https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23.pl?parameterOption=s]

* **dissolved_oxygen**: Dissolved oxygen concentration (μmol/kg) at depth. [https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23oxnu.pl?parameterOption=o]

* **log_CHL**: Log-transformed chlorophyll concentration. Proxy for phytoplankton biomass.

* **PAR(m)**: Photosynthetically Active Radiation (μE m⁻² s⁻¹).

* **log_N**: Log-transformed nitrate concentration. [https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23oxnu.pl?parameterOption=n]

* **log_P**: Log-transformed phosphate concentration. [https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/bin/woa23oxnu.pl?parameterOption=p]

* **log_Fe**: Log-transformed dissolved iron concentration at depth.[https://zenodo.org/records/6994318]

---

### 🌍 Spatial Features (Spherical Encoding)

* **coord1, coord2, coord3**
  Transformed latitude and longitude coordinates mapped onto a unit sphere:

  * Preserve spatial continuity
  * Avoid discontinuity at ±180° longitude
  * Enable distance-aware learning in ML models

---

### ⏳ Temporal Features (Cyclical Encoding)

* **time_sin, time_cos**

  Cyclical encoding of month:

  * sin(2π·month/12), cos(2π·month/12)
  * Preserve seasonality and continuity between December and January

---

## 📌 Feature Set Used in Modeling

```python
feature_cols = [
    "MLD", "salinity", "dissolved_oxygen", "T", "SST(m)",
    "log_CHL", "PAR(m)", "log_N", "log_P", "log_Fe",
    "DEPTH (m)", "coord1", "coord2", "coord3",
    "time_sin", "time_cos"
]

target_col = "log_y"
```

---

## 📌 Notes

* Log transformations are applied to reduce skewness and stabilize variance.
* Spatial and temporal encodings follow Tang et al. (2019) to preserve continuity.
* Dataset combines observational oceanographic measurements with derived features.
