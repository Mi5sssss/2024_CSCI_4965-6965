import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import date, timedelta
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
# Convert to Torch Tensors
###############################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train_numeric.values, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_bin.values, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_numeric.values, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val_bin.values, dtype=torch.float32).to(device)

###############################
# PyTorch Logistic Regression Model
###############################
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1).to(device)
    def forward(self, x):
        return self.linear(x)

def train_model(X_train_t, y_train_t, input_dim, epochs=10, lr=0.01, batch_size=64):
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Removed pin_memory=True
    model = LogisticRegressionModel(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
    return model


def predict_model(model, X_data_t):
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
# Scoring Functions
###############################
def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred)**2).mean())

def check_score(val_score, val_pred):
    val_score['pred'] = val_pred
    region_scores = []
    for region in val_score.region.unique():
        sub = val_score[val_score.region == region]
        region_rmse = rmse(sub.severity.values, sub.pred.values)
        print(f"RMSE for {region} (n={len(sub)}): {round(region_rmse,4)}")
        region_scores.append(region_rmse)
    overall_rmse = np.mean(region_scores)
    print(f"Final score: {overall_rmse}")
    return overall_rmse

###############################
# Scorecard and Summary
###############################
scores_df = pd.DataFrame()

def update_scorecard(X_train_t, y_train_t, X_val_t, y_val_bin, model_name, scores_df):
    model = train_model(X_train_t, y_train_t, X_train_t.shape[1], epochs=10)
    val_pred = predict_model(model, X_val_t)

    # Compute validation score
    val_rmse = check_score(val_score.copy(), val_pred)

    # Add to scorecard
    new_row = pd.DataFrame([{
        'Model Name': model_name,
        'Validation RMSE': val_rmse
    }])
    scores_df = pd.concat([scores_df, new_row], ignore_index=True)
    return scores_df

def summarize_scorecard(scores_df):
    best_idx = scores_df['Validation RMSE'].idxmin()
    best_row = scores_df.loc[best_idx]
    print("===== Best Model =====")
    print(f"Model Name: {best_row['Model Name']}")
    print(f"Validation RMSE: {best_row['Validation RMSE']:.4f}")
    print("=======================")

###############################
# Run and Demonstrate
###############################
scores_df = update_scorecard(X_train_t, y_train_t, X_val_t, y_val_bin, 'LogReg_PyTorch', scores_df)
print(scores_df)
summarize_scorecard(scores_df)
