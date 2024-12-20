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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import make_scorer
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import ViTModel, ViTConfig  # ViT 사용을 위한 import


# 저장 폴더 정의
#save_folder = '/content/drive/MyDrive/AE_Train(240806)/Results_Pool_CNN'
save_folder = './Results_Pool_100%_Spectrogram'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 경로 설정
accuracy_path = os.path.join(save_folder, 'transformer_accuracy.xlsx')
best_model_path = os.path.join(save_folder, 'transformer_best_model.pth')
training_loss_path = os.path.join(save_folder, 'transformer_training_loss.png')
confusion_matrix_path = os.path.join(save_folder, 'transformer_confusion_matrix.png')
hyperparameters_path = os.path.join(save_folder, 'transformer_hyperparameters.txt')
test_results_path = os.path.join(save_folder, 'transformer_test_results.xlsx')
params_vs_score_path = os.path.join(save_folder, 'transformer_params_vs_score.png')
layers_vs_score_path = os.path.join(save_folder, 'transformer_layers_vs_score.png')
"""
# 데이터셋 경로 설정
dataset_train = '/content/drive/MyDrive/AE_Train(240806)/spectrograms/dataset_signal_pool_train'
dataset_test = '/content/drive/MyDrive/AE_Train(240806)/spectrograms/dataset_signal_pool_test'
train_label = '/content/drive/MyDrive/AE_Train(240806)/spectrograms/dataset_signal_pool_train/labels.xlsx'
test_label = '/content/drive/MyDrive/AE_Train(240806)/spectrograms/dataset_signal_pool_test/labels.xlsx'
"""
# 데이터셋 경로 설정
dataset_train = './dataset_stft_pool_train_100%'
dataset_test = './dataset_stft_pool_test_100%'
train_label = './dataset_stft_pool_train_100%/labels.xlsx'
test_label = './dataset_stft_pool_test_100%/labels.xlsx'

num_epochs = 500
batch_size = 50
learning_rate = 0.0001
early_stopping_patience = 50
fold = 4

# Transformer 모델 파라미터 설정
input_dim = 3  # RGB spectrogram
d_model = 512
nhead = 8
num_layers = 4

# Update the param_grid
param_grid = {
    'learning_rate': [learning_rate],
    'batch_size': [batch_size],
    'num_epochs': [num_epochs],
    'optimizer': ['adam'],
    'd_model': [128],
    'nhead': [8], # d_model을 nhead로 나눌 수 있어야 함.
    'num_layers': [8],
    'weight_initialization': ['he']
}

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class BoilingRegimeDataset(Dataset):
    def __init__(self, excel_file, root_dir, transform=None, target_scaler=None):
        self.boiling_regime_frame = pd.read_excel(excel_file)
        self.root_dir = root_dir
        self.transform = transform

        if target_scaler is None:
            self.target_scaler = MinMaxScaler()
            self.target_scaler.fit(self.boiling_regime_frame.iloc[:, 2:4].values)
        else:
            self.target_scaler = target_scaler

    def __len__(self):
        return len(self.boiling_regime_frame)

    def __getitem__(self, idx):
        image_number = self.boiling_regime_frame.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, f'{image_number}')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        heat_flux = self.boiling_regime_frame.iloc[idx, 2].astype('float32')
        heat_transfer_coefficient = self.boiling_regime_frame.iloc[idx, 3].astype('float32')

        if self.target_scaler:
            targets = np.array([[heat_flux, heat_transfer_coefficient]])
            targets = self.target_scaler.transform(targets).flatten()
            heat_flux, heat_transfer_coefficient = targets

        boiling_regime = self.boiling_regime_frame.iloc[idx, 4]

        return image, torch.tensor(heat_flux).float(), torch.tensor(heat_transfer_coefficient).float(), torch.tensor(boiling_regime).long()


class TransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        config = ViTConfig(hidden_size=d_model, num_attention_heads=nhead, num_hidden_layers=num_layers)
        self.vit = ViTModel(config)
        self.fc_heat_flux = nn.Linear(d_model, 1)
        self.fc_heat_transfer_coefficient = nn.Linear(d_model, 1)
        self.fc_boiling_regime = nn.Linear(d_model, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        x = outputs.last_hidden_state[:, 0]  # CLS token의 출력
        heat_flux = self.fc_heat_flux(x).squeeze()
        heat_transfer_coefficient = self.fc_heat_transfer_coefficient(x).squeeze()
        boiling_regime = self.fc_boiling_regime(x)
        return heat_flux, heat_transfer_coefficient, boiling_regime


# CNN2DAlexNet
# Define the model wrapper and other necessary functions here
class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, d_model, nhead, num_layers, learning_rate=0.0001, num_epochs=30, batch_size=16, optimizer='adam', weight_initialization='default'):
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weight_initialization = weight_initialization

    def fit(self, X, y):
        self.model = TransformerModel(self.num_classes, self.d_model, self.nhead, self.num_layers).to(device)
        if self.weight_initialization == 'xavier':
            self.model.apply(self._xavier_init_weights)
        elif self.weight_initialization == 'he':
            self.model.apply(self._he_init_weights)

        criterion_regression = nn.MSELoss()
        criterion_classification = nn.CrossEntropyLoss()
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        train_loader = DataLoader(list(zip(X, y)), batch_size=self.batch_size, shuffle=True)

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
                loss_regression = criterion_regression(heat_flux, labels[:, 0].view_as(heat_flux)) + criterion_regression(heat_transfer_coefficient, labels[:, 1].view_as(heat_transfer_coefficient))
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
        self.classes_ = np.unique(y[:, 2].cpu().numpy())
        return self

    def predict(self, X):
        self.model.eval()
        all_heat_flux = []
        all_heat_transfer_coefficient = []
        all_boiling_regime = []
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(device)
            data_loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
            for features in data_loader:
                heat_flux, heat_transfer_coefficient, boiling_regime = self.model(features)
                all_heat_flux.extend(heat_flux.cpu().numpy())
                all_heat_transfer_coefficient.extend(heat_transfer_coefficient.cpu().numpy())
                _, preds = torch.max(boiling_regime, 1)
                all_boiling_regime.extend(preds.cpu().numpy())

        y_pred = np.column_stack((all_heat_flux, all_heat_transfer_coefficient, all_boiling_regime))
        return y_pred

    def score(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(device)
            data_loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
            all_preds = []
            for features in data_loader:
                _, _, boiling_regime = self.model(features)
                _, preds = torch.max(boiling_regime, 1)
                all_preds.extend(preds.cpu().numpy())
        return accuracy_score(y[:, 2].astype(int), all_preds)

    def _xavier_init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def _he_init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)


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

# Custom scorer is not needed. Use accuracy_score directly.
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

# Prepare the dataset for grid search # Cross validation 하기 때문에 validation 데이터셋 필요 없음.
# 스케일러 초기화 및 데이터셋 초기화
target_scaler = MinMaxScaler()
# 데이터셋 및 데이터 로더 초기화
full_train_dataset = BoilingRegimeDataset(excel_file=train_label, root_dir=dataset_train, transform=transform)
train_scaler = full_train_dataset.target_scaler  # 훈련 데이터의 scaler 저장
test_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test, transform=transform,
                                    target_scaler=train_scaler)

train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

features_list = []
labels_list = []

# 배치 단위로 데이터 로드
for batch in train_loader:
    images, heat_flux, heat_transfer_coefficient, boiling_regime = batch
    features_list.append(images.cpu().numpy())
    labels_list.append(torch.stack([heat_flux, heat_transfer_coefficient, boiling_regime], dim=1).cpu().numpy())
    print(f"Processed batch. Total processed: {len(features_list) * batch_size}")

# numpy 배열로 변환
features_array = np.concatenate(features_list, axis=0)
labels_array = np.concatenate(labels_list, axis=0)

print("Features shape:", features_array.shape)
print("Labels shape:", labels_array.shape)

# Prepare the dataset for grid search
# Cross validation 하기 때문에 validation 데이터셋 필요 없음.
X_train = features_array.astype(np.float32)
y_train = labels_array.astype(np.float32)
y_train[:, 2] = y_train[:, 2].astype(int)  # Ensure boiling regime is integer

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Perform grid search
num_classes = len(np.unique(labels_array[:, 2]))
model = TorchModelWrapper(num_classes=num_classes, d_model=32, nhead=2, num_layers=2)

custom_scorer = make_scorer(custom_score, greater_is_better=True)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=custom_scorer, cv=fold, verbose=2, refit=True)
grid_search.fit(X_train, y_train)

results_grid_df = analyze_results(grid_search)
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
        features = features.to(device)
        heat_flux = heat_flux.to(device)
        heat_transfer_coefficient = heat_transfer_coefficient.to(device)
        boiling_regime = boiling_regime.to(device)

        outputs_heat_flux, outputs_heat_transfer_coefficient, outputs_boiling_regime = best_model.model(features)

        # Inverse transform
        heat_flux_np = heat_flux.cpu().numpy().reshape(-1, 1)
        heat_transfer_coefficient_np = heat_transfer_coefficient.cpu().numpy().reshape(-1, 1)
        outputs_heat_flux_np = outputs_heat_flux.cpu().numpy().reshape(-1, 1)
        outputs_heat_transfer_coefficient_np = outputs_heat_transfer_coefficient.cpu().numpy().reshape(-1, 1)

        # Combine the true and predicted values for inverse scaling
        true_values = np.hstack([heat_flux_np, heat_transfer_coefficient_np])
        pred_values = np.hstack([outputs_heat_flux_np, outputs_heat_transfer_coefficient_np])

        # Ensure the scaler used for inverse transform is the same as the one used for scaling
        true_values_inverse = train_scaler.inverse_transform(true_values)
        pred_values_inverse = train_scaler.inverse_transform(pred_values)

        # Extract the inverse-transformed values
        heat_flux_true = true_values_inverse[:, 0]
        heat_transfer_true = true_values_inverse[:, 1]
        heat_flux_pred = pred_values_inverse[:, 0]
        heat_transfer_pred = pred_values_inverse[:, 1]

        _, predicted = torch.max(outputs_boiling_regime, 1)
        all_labels.extend(boiling_regime.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        all_heat_flux_true.extend(heat_flux_true)
        all_heat_flux_pred.extend(heat_flux_pred)
        all_heat_transfer_true.extend(heat_transfer_true)
        all_heat_transfer_pred.extend(heat_transfer_pred)

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

mse_heat_flux = mean_squared_error(all_heat_flux_true, all_heat_flux_pred)
mse_heat_transfer = mean_squared_error(all_heat_transfer_true, all_heat_transfer_pred)

rmspe_heat_flux = rmspe(np.array(all_heat_flux_true), np.array(all_heat_flux_pred))
rmspe_heat_transfer = rmspe(np.array(all_heat_transfer_true), np.array(all_heat_transfer_pred))

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
    class_metrics['Precision'].append(
        precision_score(np.array(all_labels) == i, np.array(all_preds) == i, zero_division=0))
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
    'Metric': ['MSE Heat Flux', 'RMSPE Heat Flux', 'MAPE Heat Flux', 'NRMSE Heat Flux', 'Q2 Heat Flux',
               'Mean P Heat Flux', 'Std. P Heat Flux',
               'MSE Heat Transfer', 'RMSPE Heat Transfer', 'MAPE Heat Transfer', 'NRMSE Heat Transfer',
               'Q2 Heat Transfer', 'Mean P Heat Transfer', 'Std. P Heat Transfer'],
    'Value': [mse_heat_flux, rmspe_heat_flux, mape_heat_flux, nrmse_heat_flux, q2_heat_flux, mean_p_heat_flux,
              std_p_heat_flux,
              mse_heat_transfer, rmspe_heat_transfer, mape_heat_transfer, nrmse_heat_transfer, q2_heat_transfer,
              mean_p_heat_transfer, std_p_heat_transfer]
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

# from google.colab import runtime

# print("All tasks completed. Disconnecting from Colab.")
# runtime.unassign()

