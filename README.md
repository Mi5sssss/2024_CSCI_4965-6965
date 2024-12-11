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

### Note
We modified `hab_functions.py` from [this repo](https://github.com/mduncan23/Predicting-Harmful-Algal-Blooms/blob/main/hab_functions.py).

## Model Training and Evaluation Workflow

### Overview
 `train_hyperparams_tuning.py` provides a complete workflow for training and evaluating multiple machine learning models (Random Forest, KNN, MLP, and Ridge Classifier) on a dataset. It includes data preprocessing, hyperparameter tuning, and evaluation metrics.

### Workflow

1. **Data Loading and Preprocessing**
- **Load Training Data**:
  - Data is read from `clean_train_data.csv` and `clean_test_data.csv`.
  - Dates are converted to ordinal format for numeric compatibility.
- **Train-Validation Split**:
  - Splits the dataset manually into training and validation sets using random shuffling.
- **Imputation and Scaling**:
  - Missing values are filled with the column mean.
  - Standard scaling is applied to normalize features.
- **PyTorch Tensor Conversion**:
  - Features and labels are converted into PyTorch tensors for compatibility with model training.

---

2. **Hyperparameter Tuning**
- **`tune_model` Function**:
  - Iterates over a grid of hyperparameters.
  - Trains models with each parameter combination.
  - Evaluates performance using RMSE and Accuracy.
  - Selects the best parameters based on evaluation results.

---

3. **Models**
#### Random Forest Classifier
- Implements a Random Forest using individual neural network trees.
- **Training**:
  - Each tree is trained independently on random bootstrap samples.
- **Prediction**:
  - Predictions are averaged across trees, with a threshold for classification.

#### KNN Classifier
- Uses distance metrics to find the k-nearest neighbors.
- **Prediction**:
  - Computes the mean label of the k-nearest neighbors and thresholds for classification.

#### MLP Classifier
- A fully connected Multi-Layer Perceptron for classification.
- **Architecture**:
  - Input layer → Hidden layers with ReLU activations → Output layer.
- **Training**:
  - Optimized using Adam and `BCEWithLogitsLoss`.
- **Prediction**:
  - Applies sigmoid activation to generate probabilities and thresholds for classification.

#### Ridge Classifier
- A linear model with L2 regularization.
- **Training**:
  - Trained using SGD with weight decay as the regularization parameter.
- **Prediction**:
  - Applies sigmoid activation for binary classification.

---

4. **Evaluation**
- **Metrics**:
  - **RMSE**: Measures the root mean squared error.
  - **Accuracy**: Calculates the proportion of correct predictions.
- **`evaluate_model` Function**:
  - Computes RMSE and Accuracy for predictions on the validation set.

---

5. **Hyperparameter Grids**
- **Random Forest**:
  - Number of estimators: `[10, 20, 50]`.
- **KNN**:
  - Number of neighbors (k): `[3, 5, 10]`.
- **MLP**:
  - Hidden layer sizes: `[(64,), (64, 64), (128, 64)]`.
  - Learning rates: `[0.01, 0.005]`.
  - Epochs: `[300]`.
- **Ridge Classifier**:
  - Regularization strength (alpha): `[0.1, 1.0, 10.0]`.
  - Learning rates: `[0.01, 0.005]`.
  - Epochs: `[300]`.
