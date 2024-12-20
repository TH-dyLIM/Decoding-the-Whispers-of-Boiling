from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import shap
import seaborn as sns
# Additional imports
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import math

# Best는 'd_model': [32], 'nhead': [4], 'num_layers': [4], 'chunk_size': [32]

# 저장 폴더 정의
save_folder = './Results_Pool_100%_AEsignal'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 경로 설정
accuracy_path = os.path.join(save_folder, 'transformer_accuracy.xlsx')
best_model_path = os.path.join(save_folder, 'transformer_best_model.pth')
training_loss_path = os.path.join(save_folder, 'transformer_training_loss.png')
confusion_matrix_path = os.path.join(save_folder, 'transformer_confusion_matrix.png')
hyperparameters_path = os.path.join(save_folder, 'transformer_hyperparameters.txt')
test_results_path = os.path.join(save_folder, 'transformer_test_results.xlsx')
params_vs_score_path = os.path.join(save_folder, 'transformer_params_vs_score2-3.png')
layers_vs_score_path = os.path.join(save_folder, 'transformer_layers_vs_score2-3.png')

# 데이터셋 경로 설정
dataset_train = './dataset_signal_pool_train_100%'
dataset_test = './dataset_signal_pool_test_100%'
train_label = './dataset_signal_pool_train_100%/labels.xlsx'
test_label = './dataset_signal_pool_test_100%/labels.xlsx'

num_epochs = 1
batch_size = 50
learning_rate = 0.0001
early_stopping_patience = 50
fold = 4

# Transformer 모델 파라미터 설정
input_dim = 1  # 1D 신호
d_model = 512
nhead = 8
num_layers = 4

# Update the param_grid
param_grid = {
    'learning_rate': [learning_rate],
    'batch_size': [batch_size],
    'num_epochs': [num_epochs],
    'optimizer': ['adam'],
    'd_model': [32],
    'nhead': [4], # d_model을 nhead로 나눌 수 있어야 함.
    'num_layers': [4], # 레이어 적을수록 좋음
    'chunk_size': [32], # 61,441 신호 데이터의 best는 64일 때.
    'weight_init': ['he']
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class BoilingRegimeDataset(Dataset):
    def __init__(self, excel_file, root_dir, feature_scaler=None, heat_flux_scaler=None, heat_transfer_scaler=None):
        self.boiling_regime_frame = pd.read_excel(excel_file)
        self.root_dir = root_dir
        self.feature_scaler = feature_scaler
        self.heat_flux_scaler = heat_flux_scaler
        self.heat_transfer_scaler = heat_transfer_scaler

    def __len__(self):
        return len(self.boiling_regime_frame)

    def __getitem__(self, idx):
        image_number = self.boiling_regime_frame.iloc[idx, 0]
        csv_file = os.path.join(self.root_dir, f'{image_number}')
        data = pd.read_csv(csv_file)
        features = data['Voltage'].values.astype('float32').reshape(1, -1)  # Reshape to (1, seq_length)

        if self.feature_scaler:
            features = self.feature_scaler.transform(features).flatten()
        features = torch.tensor(features).float().to(device)  # shape: (61441,)

        heat_flux = self.boiling_regime_frame.iloc[idx, 2].astype('float32')
        heat_transfer_coefficient = self.boiling_regime_frame.iloc[idx, 3].astype('float32')

        if self.heat_flux_scaler:
            heat_flux = self.heat_flux_scaler.transform([[heat_flux]])[0][0]
        if self.heat_transfer_scaler:
            heat_transfer_coefficient = self.heat_transfer_scaler.transform([[heat_transfer_coefficient]])[0][0]

        boiling_regime = self.boiling_regime_frame.iloc[idx, 4]

        heat_flux = torch.tensor(heat_flux).float().to(device)
        heat_transfer_coefficient = torch.tensor(heat_transfer_coefficient).float().to(device)
        boiling_regime = torch.tensor(boiling_regime).long().to(device)

        return features, heat_flux, heat_transfer_coefficient, boiling_regime


# 스케일러 초기화 및 데이터셋 초기화
feature_scaler = StandardScaler()
heat_flux_scaler = MinMaxScaler()
heat_transfer_scaler = MinMaxScaler()

full_train_dataset = BoilingRegimeDataset(excel_file=train_label, root_dir=dataset_train)
features_list = []
heat_flux_list = []
heat_transfer_list = []
labels_list = []

for i in range(len(full_train_dataset)):
    image_number = full_train_dataset.boiling_regime_frame.iloc[i, 0]
    csv_file = os.path.join(dataset_train, f'{image_number}')
    data = pd.read_csv(csv_file)

    features = data['Voltage'].values.astype('float32').reshape(1, -1)
    features_list.append(features)

    heat_flux = full_train_dataset.boiling_regime_frame.iloc[i, 2].astype('float32')
    heat_transfer_coefficient = full_train_dataset.boiling_regime_frame.iloc[i, 3].astype('float32')
    boiling_regime = full_train_dataset.boiling_regime_frame.iloc[i, 4]
    heat_flux_list.append([heat_flux])
    heat_transfer_list.append([heat_transfer_coefficient])
    labels_list.append([heat_flux, heat_transfer_coefficient, boiling_regime])
    print(i)

features_array = np.vstack(features_list)
heat_flux_array = np.array(heat_flux_list)
heat_transfer_array = np.array(heat_transfer_list)
labels_array = np.array(labels_list)

# 스케일러를 사용하여 데이터 스케일링
features_array = feature_scaler.fit_transform(features_array)
heat_flux_array = heat_flux_scaler.fit_transform(heat_flux_array)
heat_transfer_array = heat_transfer_scaler.fit_transform(heat_transfer_array)
labels_array[:, 0] = heat_flux_array.flatten()
labels_array[:, 1] = heat_transfer_array.flatten()

features_array = np.nan_to_num(features_array)

full_train_dataset = BoilingRegimeDataset(excel_file=train_label, root_dir=dataset_train, feature_scaler=feature_scaler,
                                          heat_flux_scaler=heat_flux_scaler, heat_transfer_scaler=heat_transfer_scaler)
test_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test, feature_scaler=feature_scaler,
                                    heat_flux_scaler=heat_flux_scaler, heat_transfer_scaler=heat_transfer_scaler)

train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, chunk_size=1024):
        super().__init__()
        self.model_type = 'Transformer'
        self.chunk_size = chunk_size
        self.stride = chunk_size // 2  # Automatically set stride to half of chunk_size
        self.input_proj = nn.Linear(chunk_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.decoder_heat_flux = nn.Linear(d_model, 1)
        self.decoder_heat_transfer = nn.Linear(d_model, 1)
        self.decoder_boiling_regime = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.zeros_(p)

    def forward(self, src):
        # src shape: (batch_size, 1, 61441)
        batch_size = src.shape[0]

        # Add a dimension if necessary
        if src.dim() == 2:
            src = src.unsqueeze(1)  # shape: (batch_size, 1, 61441)

        # Unfold the input into overlapping chunks
        chunks = src.unfold(2, self.chunk_size, self.stride)  # shape: (batch_size, 1, num_chunks, chunk_size)
        chunks = chunks.squeeze(1)  # shape: (batch_size, num_chunks, chunk_size)

        # Project chunks to d_model dimension
        chunks = self.input_proj(chunks)  # shape: (batch_size, num_chunks, d_model)

        # Add positional encoding
        chunks = self.pos_encoder(chunks.transpose(0, 1)).transpose(0, 1)

        # Pass through transformer
        output = self.transformer_encoder(chunks.transpose(0, 1)).transpose(0, 1)

        # Global average pooling
        output = output.mean(dim=1)  # shape: (batch_size, d_model)

        # Decode outputs
        heat_flux = self.decoder_heat_flux(output).squeeze(-1)
        heat_transfer = self.decoder_heat_transfer(output).squeeze(-1)
        boiling_regime = self.decoder_boiling_regime(output)

        return heat_flux, heat_transfer, boiling_regime

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes,
                 learning_rate=0.01, num_epochs=50, batch_size=16, optimizer='adam',
                 chunk_size=1024, weight_init='he'):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.chunk_size = chunk_size
        self.stride = chunk_size // 2
        self.weight_init = weight_init

    def fit(self, X, y):
        self.model = TransformerModel(self.input_dim, self.d_model, self.nhead, self.num_layers,
                                      self.num_classes, self.chunk_size).to(device)

        if self.weight_init == 'he':
            self.model.apply(self._he_init_weights)
        elif self.weight_init == 'xavier':
            self.model.apply(self._xavier_init_weights)

        criterion_regression = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

        train_loader = DataLoader(list(zip(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))),
                                  batch_size=self.batch_size, shuffle=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                heat_flux, heat_transfer_coefficient, boiling_regime = self.model(features)
                loss_regression = criterion_regression(heat_flux, labels[:, 0]) + \
                                  criterion_regression(heat_transfer_coefficient, labels[:, 1])
                loss_classification = criterion_classification(boiling_regime, labels[:, 2].long())
                loss = loss_regression + loss_classification
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        self.model.load_state_dict(best_model_wts)
        self.classes_ = np.unique(y[:, 2])
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            data_loader = DataLoader(torch.tensor(X, dtype=torch.float32),
                                     batch_size=self.batch_size, shuffle=False)
            all_preds = []
            for features in data_loader:
                features = features.to(device)
                heat_flux, heat_transfer, boiling_regime = self.model(features)
                _, boiling_preds = torch.max(boiling_regime, 1)
                preds = torch.cat([heat_flux.unsqueeze(1),
                                   heat_transfer.unsqueeze(1),
                                   boiling_preds.float().unsqueeze(1)], dim=1)
                all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)

    def score(self, X, y):
        self.model.eval()
        with torch.no_grad():
            data_loader = DataLoader(torch.tensor(X, dtype=torch.float32).to(device), batch_size=self.batch_size,
                                     shuffle=False)
            all_preds = []
            for features in data_loader:
                _, _, boiling_regime = self.model(features)
                _, preds = torch.max(boiling_regime, 1)
                all_preds.extend(preds.cpu().numpy())
        return accuracy_score(y[:, 2].astype(int), all_preds)  # Only use boiling regime for accuracy

    def _xavier_init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def _he_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

# Function to analyze grid search results
def analyze_results(grid_search):
    results = []
    for params, mean_score, scores in zip(grid_search.cv_results_['params'],
                                          grid_search.cv_results_['mean_test_score'],
                                          grid_search.cv_results_['std_test_score']):
        results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': scores
        })
    return pd.DataFrame(results)

# Prepare the dataset for grid search
# labels_array = full_train_dataset.boiling_regime_frame.iloc[:, [2, 3, 4]].values  # heat_flux, heat_transfer_coefficient, boiling_regime
# Cross validation 하기 때문에 validation 데이터셋 필요 없음.

X_train = features_array
y_train = labels_array

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
y_train[:, 2] = y_train[:, 2].astype(int)  # Ensure boiling regime is integer

# Perform grid search
num_classes = len(np.unique(labels_array[:, 2]))
model = TorchModelWrapper(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                          num_classes=num_classes)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

from sklearn.metrics import make_scorer

def rmspe(y_true, y_pred):
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.max() - y_true.min())

def custom_score(y_true, y_pred):
    # Heat flux NRMSE
    nrmse_heat_flux = nrmse(y_true[:, 0], y_pred[:, 0])

    # Heat transfer coefficient NRMSE
    nrmse_heat_transfer = nrmse(y_true[:, 1], y_pred[:, 1])

    # Boiling regime accuracy
    accuracy_boiling = accuracy_score(y_true[:, 2].astype(int), y_pred[:, 2].astype(int))

    # Calculate accuracy from NRMSE
    accuracy_heat_flux = 1 - nrmse_heat_flux
    accuracy_heat_transfer = 1 - nrmse_heat_transfer

    # Combine scores (you can adjust weights as needed)
    combined_score = (accuracy_heat_flux / 3 +
                      accuracy_heat_transfer / 3 +
                      accuracy_boiling / 3)

    return combined_score



custom_scorer = make_scorer(custom_score, greater_is_better=True)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=custom_scorer, cv=fold, verbose=2, refit=True)
grid_search.fit(X_train, y_train)

# Analyze results
results_grid_df = analyze_results(grid_search)

# Save the best model
best_model = grid_search.best_estimator_
torch.save(best_model.model.state_dict(), best_model_path)

# Save hyperparameters
with open(hyperparameters_path, 'w') as f:
    f.write(f'Best parameters: {grid_search.best_params_}\n')

# Test the best model
best_model.model.eval()
all_labels = []
all_preds = []
all_heat_flux_true = []
all_heat_flux_pred = []
all_heat_transfer_true = []
all_heat_transfer_pred = []

with torch.no_grad():
    for features, heat_flux, heat_transfer_coefficient, boiling_regime in test_loader:
        features = np.nan_to_num(features.cpu().numpy())
        features = features.to(device)
        heat_flux = heat_flux.to(device)
        heat_transfer_coefficient = heat_transfer_coefficient.to(device)
        boiling_regime = boiling_regime.to(device)
        outputs_heat_flux, outputs_heat_transfer_coefficient, outputs_boiling_regime = best_model.model(features)

        heat_flux_np = heat_flux.cpu().numpy().reshape(-1, 1)
        heat_transfer_coefficient_np = heat_transfer_coefficient.cpu().numpy().reshape(-1, 1)
        outputs_heat_flux_np = outputs_heat_flux.cpu().numpy().reshape(-1, 1)
        outputs_heat_transfer_coefficient_np = outputs_heat_transfer_coefficient.cpu().numpy().reshape(-1, 1)

        heat_flux_true = heat_flux_scaler.inverse_transform(heat_flux_np).flatten()
        heat_transfer_true = heat_transfer_scaler.inverse_transform(heat_transfer_coefficient_np).flatten()
        heat_flux_pred = heat_flux_scaler.inverse_transform(outputs_heat_flux_np).flatten()
        heat_transfer_pred = heat_transfer_scaler.inverse_transform(outputs_heat_transfer_coefficient_np).flatten()

        _, predicted = torch.max(outputs_boiling_regime, 1)
        all_labels.extend(boiling_regime.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        all_heat_flux_true.extend(heat_flux_true)
        all_heat_flux_pred.extend(heat_flux_pred)
        all_heat_transfer_true.extend(heat_transfer_true)
        all_heat_transfer_pred.extend(heat_transfer_pred)

mse_heat_flux = mean_squared_error(all_heat_flux_true, all_heat_flux_pred)
mse_heat_transfer = mean_squared_error(all_heat_transfer_true, all_heat_transfer_pred)


rmspe_heat_flux = rmspe(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))
rmspe_heat_transfer = rmspe(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))

def mean_p(y_true, y_pred):
    return np.mean(y_true - y_pred)

def std_p(y_true, y_pred):
    return np.std(y_true - y_pred)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def q2(y_true, y_pred):
    return r2_score(y_true, y_pred)

mean_p_heat_flux = mean_p(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))
std_p_heat_flux = std_p(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))
mape_heat_flux = mape(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))
nrmse_heat_flux = nrmse(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))
q2_heat_flux = q2(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))

mean_p_heat_transfer = mean_p(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))
std_p_heat_transfer = std_p(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))
mape_heat_transfer = mape(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))
nrmse_heat_transfer = nrmse(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))
q2_heat_transfer = q2(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))

cm = confusion_matrix(all_labels, all_preds)
df_cm = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))

class_metrics = {
    'Class': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for i in range(num_classes):
    class_metrics['Class'].append(i)
    class_metrics['Accuracy'].append(accuracy_score(np.array(all_labels) == i, np.array(all_preds) == i))
    class_metrics['Precision'].append(precision_score(np.array(all_labels) == i, np.array(all_preds) == i, zero_division=0))
    class_metrics['Recall'].append(recall_score(np.array(all_labels) == i, np.array(all_preds) == i, zero_division=0))
    class_metrics['F1 Score'].append(f1_score(np.array(all_labels) == i, np.array(all_preds) == i, zero_division=0))

results_df = pd.DataFrame({
    'True Heat Flux': all_heat_flux_true,
    'Predicted Heat Flux': all_heat_flux_pred,
    'True Heat Transfer Coefficient': all_heat_transfer_true,
    'Predicted Heat Transfer Coefficient': all_heat_transfer_pred,
    'True Boiling Regime': all_labels,
    'Predicted Boiling Regime': all_preds
})

metrics_df = pd.DataFrame({
    'Metric': ['MSE Heat Flux', 'RMSPE Heat Flux', 'MAPE Heat Flux', 'NRMSE Heat Flux', 'Q2 Heat Flux', 'Mean P Heat Flux', 'Std. P Heat Flux',
               'MSE Heat Transfer', 'RMSPE Heat Transfer', 'MAPE Heat Transfer', 'NRMSE Heat Transfer', 'Q2 Heat Transfer', 'Mean P Heat Transfer', 'Std. P Heat Transfer'],
    'Value': [mse_heat_flux, rmspe_heat_flux, mape_heat_flux, nrmse_heat_flux, q2_heat_flux, mean_p_heat_flux, std_p_heat_flux,
              mse_heat_transfer, rmspe_heat_transfer, mape_heat_transfer, nrmse_heat_transfer, q2_heat_transfer, mean_p_heat_transfer, std_p_heat_transfer]
})

class_accuracies_df = pd.DataFrame(class_metrics)

with pd.ExcelWriter(test_results_path) as writer:
    results_df.to_excel(writer, sheet_name='Predictions', index=False)
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    class_accuracies_df.to_excel(writer, sheet_name='Class Metrics', index=False)
    results_grid_df.to_excel(writer, sheet_name='Grid', index=False)

print(f'Test results saved to {test_results_path}')
print(f'Test Heat Flux MSE: {mse_heat_flux}')
print(f'Test Heat Transfer Coefficient MSE: {mse_heat_transfer}')
print(f'Test Heat Flux RMSPE: {rmspe_heat_flux}')
print(f'Test Heat Transfer Coefficient RMSPE: {rmspe_heat_transfer}')
print(f'Test Heat Flux MAPE: {mape_heat_flux}')
print(f'Test Heat Transfer Coefficient MAPE: {mape_heat_transfer}')
print(f'Test Heat Flux NRMSE: {nrmse_heat_flux}')
print(f'Test Heat Transfer Coefficient NRMSE: {nrmse_heat_transfer}')
print(f'Test Heat Flux Q2: {q2_heat_flux}')
print(f'Test Heat Transfer Coefficient Q2: {q2_heat_transfer}')
print(f'Test Heat Flux Mean P: {mean_p_heat_flux}')
print(f'Test Heat Transfer Coefficient Mean P: {mean_p_heat_transfer}')
print(f'Test Heat Flux Std. P: {std_p_heat_flux}')
print(f'Test Heat Transfer Coefficient Std. P: {std_p_heat_transfer}')

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.set(style="whitegrid")
cmap = sns.light_palette("navy", as_cmap=True)
sns.heatmap(df_cm, annot=True, fmt='g', cmap=cmap,
            linewidths=0.5, cbar=True, square=True)
plt.title('Confusion Matrix', fontsize=20, pad=20)
plt.xlabel('Predicted', fontsize=15, labelpad=10)
plt.ylabel('True', fontsize=15, labelpad=10)
plt.tick_params(axis='both', which='major', labelsize=12)
for _, spine in plt.gca().spines.items():
    spine.set_visible(True)
    spine.set_color('navy')
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
