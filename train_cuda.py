import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import date, timedelta
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.svm import SVC as cuSVC
from cuml.linear_model import RidgeClassifier as cuRidge
from cuml.neural_network import MLPClassifier as cuMLP
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
sns.set_theme()

###############################
# Load and Preprocess Data
###############################
# Load training data
full_train = pd.read_csv('./Data/clean_train_data.csv')
full_train.drop('Unnamed: 0', axis=1, inplace=True)

# Convert ordinal integers to date objects
full_train['date'] = full_train['date'].apply(lambda x: date.fromordinal(int(x)) if not pd.isnull(x) else x)
full_train['img_date'] = full_train['img_date'].apply(lambda x: date.fromordinal(int(x)) if not pd.isnull(x) else x)

# Import hab_functions and process important info
import hab_functions

# Ensure timedelta operations in hab_functions work with datetime.date
hab_functions.get_important_info(full_train)

# Load test data
full_test_df = pd.read_csv('./Data/clean_test_data.csv')
full_test_df.drop('Unnamed: 0', axis=1, inplace=True)
full_test_df['date'] = full_test_df['date'].apply(lambda x: date.fromordinal(int(x)) if not pd.isnull(x) else x)
full_test_df['img_date'] = full_test_df['img_date'].apply(lambda x: date.fromordinal(int(x)) if not pd.isnull(x) else x)

# Separate features and labels
X = full_train.drop(['severity', 'uid', 'region'], axis=1)
y = full_train['severity']
score = full_train[['uid', 'region']]

###############################
# Manual Train-Validation Split
###############################
rnd = np.random.RandomState(42)
indices = rnd.permutation(len(X))
test_size = 0.276
split_idx = int(len(X)*(1 - test_size))
train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()
y_val = y.iloc[val_idx].copy()

# Name the severity column so join works properly
y_val = y_val.rename('severity')
val_score = score.iloc[val_idx].join(y_val, how='right')

# severity {1,2} -> {0,1}
y_train_bin = y_train.map(lambda x: 0 if x == 1 else 1)
y_val_bin = y_val.map(lambda x: 0 if x == 1 else 1)

###############################
# Impute and Scale
###############################
# Identify non-numeric columns
non_numeric_cols = X_train.select_dtypes(include=['object', 'datetime']).columns

# Separate numeric and non-numeric columns
X_train_numeric = X_train.drop(non_numeric_cols, axis=1)
X_val_numeric = X_val.drop(non_numeric_cols, axis=1)

# Mean imputation for numeric columns
X_train_numeric = X_train_numeric.fillna(X_train_numeric.mean())
X_val_numeric = X_val_numeric.fillna(X_train_numeric.mean())  # Use train means for validation

# Add back non-numeric columns without changes
X_train = pd.concat([X_train_numeric, X_train[non_numeric_cols]], axis=1)
X_val = pd.concat([X_val_numeric, X_val[non_numeric_cols]], axis=1)

# Standard scaling for numeric columns
train_mean = X_train_numeric.mean()
train_std = X_train_numeric.std(ddof=0).replace(0, 1)  # Avoid division by zero
X_train_numeric = (X_train_numeric - train_mean) / train_std
X_val_numeric = (X_val_numeric - train_mean) / train_std

# Update scaled values back into X_train and X_val
X_train.update(X_train_numeric)
X_val.update(X_val_numeric)

###############################
# Logistic Regression Model (PyTorch)
###############################
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1).to(device)
    def forward(self, x):
        return self.linear(x)

def train_logistic_regression(X_train_t, y_train_t, input_dim, epochs=10, lr=0.01, batch_size=64):
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LogisticRegressionModel(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
    return model

def predict_logistic_regression(model, X_data_t):
    model.eval()
    with torch.no_grad():
        X_data_t = X_data_t.to(device)
        outputs = model(X_data_t).view(-1)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long().cpu().numpy()
    # Map 0->1 and 1->2 back
    preds_original = np.where(preds == 0, 1, 2)
    return preds_original

###############################
# GPU-Compatible Models
###############################
def train_random_forest(X_train, y_train):
    model = cuRF(n_estimators=100, random_state=42)
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    model = cuKNN(n_neighbors=n_neighbors)
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    return model

def train_mlp(X_train, y_train):
    model = cuMLP(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    return model

def train_svm(X_train, y_train):
    model = cuSVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    return model

def train_ridge(X_train, y_train):
    model = cuRidge()
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    return model

def evaluate_model(model, X_val, y_val, is_pytorch=False):
    if is_pytorch:
        preds = predict_logistic_regression(model, torch.tensor(X_val.values, dtype=torch.float32).to(device))
    else:
        preds = model.predict(X_val.astype(np.float32))
    rmse_score = mean_squared_error(y_val, preds, squared=False)
    acc_score = accuracy_score(y_val, preds)
    print(f"Model: {type(model).__name__ if not is_pytorch else 'LogisticRegressionModel'}, RMSE: {rmse_score:.4f}, Accuracy: {acc_score:.4f}")
    return rmse_score, acc_score

###############################
# Evaluate All Models with GPU Support
###############################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train_numeric.values, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_bin.values, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_numeric.values, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_bin.values, dtype=torch.float32).to(device)

models = {
    "Logistic Regression": lambda X_train, y_train: train_logistic_regression(X_train_t, y_train_t, X_train_t.shape[1], epochs=10),
    "Random Forest": train_random_forest,
    "KNN": train_knn,
    "MLP": train_mlp,
    "SVM": train_svm,
    "Ridge": train_ridge
}
scores = {}
hyperparameters = {}

for model_name, train_func in models.items():
    print(f"Training {model_name}...")
    if model_name == "Logistic Regression":
        model = train_func(X_train_numeric, y_train_bin)
        rmse, acc = evaluate_model(model, X_val_numeric, y_val_bin, is_pytorch=True)
        scores[model_name] = {"RMSE": rmse, "Accuracy": acc}
        hyperparameters[model_name] = {"epochs": 10, "lr": 0.01, "batch_size": 64}
    elif model_name == "KNN":
        model = train_func(X_train_numeric, y_train, n_neighbors=5)
        rmse, acc = evaluate_model(model, X_val_numeric, y_val)
        scores[model_name] = {"RMSE": rmse, "Accuracy": acc}
        hyperparameters[model_name] = {"n_neighbors": 5}
    elif model_name == "MLP":
        model = train_func(X_train_numeric, y_train)
        rmse, acc = evaluate_model(model, X_val_numeric, y_val)
        scores[model_name] = {"RMSE": rmse, "Accuracy": acc}
        hyperparameters[model_name] = {"hidden_layer_sizes": (64, 64), "max_iter": 500}
    elif model_name == "SVM":
        model = train_func(X_train_numeric, y_train)
        rmse, acc = evaluate_model(model, X_val_numeric, y_val)
        scores[model_name] = {"RMSE": rmse, "Accuracy": acc}
        hyperparameters[model_name] = {"kernel": 'rbf', "probability": True}
    elif model_name == "Ridge":
        model = train_func(X_train_numeric, y_train)
        rmse, acc = evaluate_model(model, X_val_numeric, y_val)
        scores[model_name] = {"RMSE": rmse, "Accuracy": acc}
        hyperparameters[model_name] = {"solver": "auto"}
    elif model_name == "Random Forest":
        model = train_func(X_train_numeric, y_train)
        rmse, acc = evaluate_model(model, X_val_numeric, y_val)
        scores[model_name] = {"RMSE": rmse, "Accuracy": acc}
        hyperparameters[model_name] = {"n_estimators": 100}

# Display Results
print("\nModel Performance:")
for model_name, metrics in scores.items():
    print(f"{model_name}: RMSE={metrics['RMSE']:.4f}, Accuracy={metrics['Accuracy']:.4f}, Hyperparameters={hyperparameters[model_name]}")
