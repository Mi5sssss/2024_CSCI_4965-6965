# 2024_CSCI_4965-6965
2024 Fall AI for Conservation

## Satellite Data Processing Pipeline

### Overview
Processes satellite metadata, integrates climate and elevation data, and extracts features for modeling(`data_processing.py`, `hab_functions.py`).

### Dependencies
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `geopandas`, `shapely`, `geopy`
- `planetary_computer`, `pystac_client`
- `rioxarray`, `odc.stac`, `pygrib`
- `scikit-learn`

### Workflow
1. **Data Preparation**:
   - Read `metadata.csv` and `train_labels.csv`.
   - Add seasonal information based on months.
2. **Geospatial Operations**:
   - Convert metadata to GeoDataFrames.
   - Generate rolling averages of regional severity.
3. **Satellite Data Retrieval**:
   - Query Sentinel/Landsat data using bounding boxes and date ranges.
   - Filter images by cloud cover and location.
4. **Raster Data Processing**:
   - Clip images to bounding boxes.
   - Extract RGB statistics (`mean`, `median`, etc.).
5. **Elevation Integration**:
   - Retrieve Copernicus DEM data and compute average elevation.
6. **Climate Data Integration**:
   - Fetch NOAA HRRR temperature data.
7. **Feature Set**:
   - Combine RGB, elevation, and climate features.
   - One-hot encode seasons.

### Outputs
- Final DataFrame (`model_df`) with features:
  - RGB (`red_mean`, `green_median`, etc.)
  - Elevation and climate data
  - Encoded seasons and metadata (`severity`, coordinates).
