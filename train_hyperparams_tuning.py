import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')
sns.set_theme()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################
# Load and Preprocess Data
###############################
# Load training data
full_train = pd.read_csv('./Data/clean_train_data.csv')
full_train.drop('Unnamed: 0', axis=1, inplace=True)

full_train['date'] = full_train['date'].apply(lambda x: x.toordinal() if isinstance(x, date) else x)
full_train['img_date'] = full_train['img_date'].apply(lambda x: x.toordinal() if isinstance(x, date) else x)

# Load test data
full_test_df = pd.read_csv('./Data/clean_test_data.csv')
full_test_df.drop('Unnamed: 0', axis=1, inplace=True)
full_test_df['date'] = full_test_df['date'].apply(lambda x: date.fromordinal(int(x)) if not pd.isnull(x) else x)
full_test_df['img_date'] = full_test_df['img_date'].apply(lambda x: date.fromordinal(int(x)) if not pd.isnull(x) else x)

# Separate features and labels
X = full_train.drop(['severity', 'uid', 'region'], axis=1)
y = full_train['severity']

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

# severity {1,2} -> {0,1}
y_train_bin = y_train.map(lambda x: 0 if x == 1 else 1)
y_val_bin = y_val.map(lambda x: 0 if x == 1 else 1)

###############################
# Impute and Scale
###############################
# Mean imputation
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_train.mean())

# Standard scaling
train_mean = X_train.mean()
train_std = X_train.std(ddof=0).replace(0, 1)  # Avoid division by zero
X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_bin.values, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_bin.values, dtype=torch.float32).to(device)

###############################
# Hyperparameter Tuning
###############################
def tune_model(train_func, predict_func, param_grid, X_train_t, y_train_t, X_val_t, y_val_t):
    best_rmse, best_acc = float('inf'), 0
    best_params = {}

    for params in param_grid:
        print(f"Testing parameters: {params}")
        model = train_func(X_train_t, y_train_t, **params)
        preds = predict_func(model, X_val_t)
        rmse, acc = evaluate_model(preds, y_val_t)
        print(f"RMSE: {rmse:.4f}, Accuracy: {acc:.4f}")

        if rmse < best_rmse:
            best_rmse, best_acc, best_params = rmse, acc, params

    print(f"Best Params: {best_params}, Best RMSE: {best_rmse:.4f}, Best Accuracy: {best_acc:.4f}")
    return best_params, best_rmse, best_acc

###############################
# Random Forest Classifier
###############################
def train_random_forest(X_train_t, y_train_t, n_estimators=10):
    trees = []
    for _ in range(n_estimators):
        indices = torch.randint(0, len(X_train_t), (len(X_train_t),))
        sample_X, sample_y = X_train_t[indices], y_train_t[indices]
        tree = nn.Linear(sample_X.size(1), 1).to(device)
        optimizer = optim.Adam(tree.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(50):  # Training each tree
            optimizer.zero_grad()
            preds = tree(sample_X).view(-1)
            loss = criterion(preds, sample_y)
            loss.backward()
            optimizer.step()
        trees.append(tree)
    return trees

def predict_random_forest(trees, X_val_t):
    outputs = torch.zeros((len(X_val_t), len(trees))).to(device)
    for i, tree in enumerate(trees):
        outputs[:, i] = torch.sigmoid(tree(X_val_t).view(-1))
    return (outputs.mean(dim=1) >= 0.5).long()

###############################
# KNN Classifier
###############################
def predict_knn(X_train_t, y_train_t, X_val_t, k=5):
    distances = torch.cdist(X_val_t, X_train_t)
    knn_indices = torch.topk(distances, k, largest=False).indices
    preds = y_train_t[knn_indices].mean(dim=1) >= 0.5
    return preds.long()

###############################
# MLP Classifier
###############################
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train_t, y_train_t, input_dim, hidden_sizes=(64, 64), epochs=50, lr=0.01):
    model = MLPModel(input_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t).view(-1)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    return model

def predict_mlp(model, X_val_t):
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_val_t).view(-1)) >= 0.5
    return preds.long()

###############################
# Ridge Classifier
###############################
class RidgeClassifier(nn.Module):
    def __init__(self, input_dim, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

def train_ridge(X_train_t, y_train_t, input_dim, alpha=1.0, epochs=50, lr=0.01):
    model = RidgeClassifier(input_dim, alpha).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=alpha)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t).view(-1)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    return model

def predict_ridge(model, X_val_t):
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_val_t).view(-1)) >= 0.5
    return preds.long()

###############################
# Evaluate Models
###############################
def evaluate_model(preds, y_val_t):
    rmse_score = mean_squared_error(y_val_t.cpu(), preds.cpu(), squared=False)
    acc_score = accuracy_score(y_val_t.cpu(), preds.cpu())
    return rmse_score, acc_score

###############################
# Run All Models
###############################
# Random Forest
rf_params = [{'n_estimators': n} for n in [10, 20, 50]]  # Increased n_estimators for more trees
best_rf_params, best_rf_rmse, best_rf_acc = tune_model(
    train_random_forest, predict_random_forest, rf_params, X_train_t, y_train_t, X_val_t, y_val_t
)

# KNN
knn_params = [{'k': k} for k in [3, 5, 10]]  # Increased maximum k value

def train_knn(X_train_t, y_train_t, k=5):
    return {'X_train_t': X_train_t, 'y_train_t': y_train_t, 'k': k}  # Dummy for KNN, no actual training

def predict_knn_wrapper(model, X_val_t):
    return predict_knn(model['X_train_t'], model['y_train_t'], X_val_t, model['k'])

best_knn_params, best_knn_rmse, best_knn_acc = tune_model(
    train_knn, predict_knn_wrapper, knn_params, X_train_t, y_train_t, X_val_t, y_val_t
)

# MLP
mlp_params = [
    {'hidden_sizes': sizes, 'epochs': epochs, 'lr': lr}
    for sizes in [(64,), (64, 64), (128, 64)]
    for epochs in [300]  # Increased epochs for more training steps
    for lr in [0.01, 0.005]
]
best_mlp_params, best_mlp_rmse, best_mlp_acc = tune_model(
    lambda X_train_t, y_train_t, hidden_sizes, epochs, lr: train_mlp(
        X_train_t, y_train_t, X_train_t.size(1), hidden_sizes, epochs, lr
    ),
    predict_mlp,
    mlp_params,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t
)

# Ridge
ridge_params = [
    {'alpha': alpha, 'epochs': epochs, 'lr': lr}
    for alpha in [0.1, 1.0, 10.0]
    for epochs in [300]  # Increased epochs for ridge training
    for lr in [0.01, 0.005]
]
best_ridge_params, best_ridge_rmse, best_ridge_acc = tune_model(
    lambda X_train_t, y_train_t, alpha, epochs, lr: train_ridge(
        X_train_t, y_train_t, X_train_t.size(1), alpha, epochs, lr
    ),
    predict_ridge,
    ridge_params,
    X_train_t,
    y_train_t,
    X_val_t,
    y_val_t
)

###############################
# Display Results
###############################
print("\nBest Model Performances:")
print(f"Random Forest - Best Params: {best_rf_params}, RMSE: {best_rf_rmse:.4f}, Accuracy: {best_rf_acc:.4f}")
print(f"KNN - Best Params: {best_knn_params}, RMSE: {best_knn_rmse:.4f}, Accuracy: {best_knn_acc:.4f}")
print(f"MLP - Best Params: {best_mlp_params}, RMSE: {best_mlp_rmse:.4f}, Accuracy: {best_mlp_acc:.4f}")
print(f"Ridge - Best Params: {best_ridge_params}, RMSE: {best_ridge_rmse:.4f}, Accuracy: {best_ridge_acc:.4f}")