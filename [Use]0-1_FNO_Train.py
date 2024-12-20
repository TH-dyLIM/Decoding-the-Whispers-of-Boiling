import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import torch.fft
from sklearn.metrics import make_scorer


# 저장 폴더 정의
save_folder = './Results_Pool_100%_AEsignal_FNO'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 경로 설정 (필요에 따라 수정)
accuracy_path = os.path.join(save_folder, 'fno_optimized_accuracy.xlsx')
best_model_path = os.path.join(save_folder, 'fno_optimized_best_model.pth')
training_loss_path = os.path.join(save_folder, 'fno_optimized_training_loss.png')
confusion_matrix_path = os.path.join(save_folder, 'fno_optimized_confusion_matrix.png')
hyperparameters_path = os.path.join(save_folder, 'fno_optimized_hyperparameters.txt')
test_results_path = os.path.join(save_folder, 'fno_optimized_test_results.xlsx')

# 데이터셋 경로 설정
dataset_train = './dataset_signal_pool_train_100%'
dataset_test = './dataset_signal_pool_test_100%'
train_label = os.path.join(dataset_train, 'labels.xlsx')
test_label = os.path.join(dataset_test, 'labels.xlsx')

num_epochs = 1
batch_size = 50
learning_rate = 0.0001
early_stopping_patience = 50
fold = 4  # K-Fold 교차 검증을 위한 fold 수

# 파라미터 그리드 설정
param_grid = {
    'modes': [1000, 2000, 3687],
    'width': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'learning_rate': [learning_rate],
    'batch_size': [batch_size],
    'num_epochs': [num_epochs],
    'early_stopping_patience': [early_stopping_patience]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class BoilingRegimeDataset(Dataset):
    def __init__(self, excel_file, root_dir, feature_scaler=None, target_scaler=None):
        self.boiling_regime_frame = pd.read_excel(excel_file)
        self.root_dir = root_dir
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def __len__(self):
        return len(self.boiling_regime_frame)

    def __getitem__(self, idx):
        image_number = self.boiling_regime_frame.iloc[idx, 0]
        csv_file = os.path.join(self.root_dir, f'{image_number}')
        data = pd.read_csv(csv_file)
        features = data['Voltage'].values.astype('float32').reshape(-1)

        if self.feature_scaler:
            features = self.feature_scaler.transform([features]).flatten()
        features = torch.tensor(features).unsqueeze(0).to(device)  # (1, N)

        heat_flux = self.boiling_regime_frame.iloc[idx, 2].astype('float32')
        heat_transfer_coefficient = self.boiling_regime_frame.iloc[idx, 3].astype('float32')

        if self.target_scaler:
            targets = np.array([[heat_flux, heat_transfer_coefficient]])
            targets = self.target_scaler.transform(targets).flatten()
            heat_flux, heat_transfer_coefficient = targets

        boiling_regime = self.boiling_regime_frame.iloc[idx, 4]

        heat_flux = torch.tensor(heat_flux).float().to(device)
        heat_transfer_coefficient = torch.tensor(heat_transfer_coefficient).float().to(device)
        boiling_regime = torch.tensor(boiling_regime).long().to(device)

        return features, heat_flux, heat_transfer_coefficient, boiling_regime

# 스케일러 초기화
feature_scaler = StandardScaler()
target_scaler = MinMaxScaler()

# 훈련 데이터 로드 및 스케일러 학습
def load_features_and_targets(dataset):
    features_list = []
    targets_list = []
    for idx in range(len(dataset.boiling_regime_frame)):
        image_number = dataset.boiling_regime_frame.iloc[idx, 0]
        csv_file = os.path.join(dataset.root_dir, f'{image_number}')
        data = pd.read_csv(csv_file)
        features = data['Voltage'].values.astype('float32')
        features_list.append(features)
        heat_flux = dataset.boiling_regime_frame.iloc[idx, 2].astype('float32')
        heat_transfer_coefficient = dataset.boiling_regime_frame.iloc[idx, 3].astype('float32')
        targets_list.append([heat_flux, heat_transfer_coefficient])
    return np.array(features_list), np.array(targets_list)

# 임시 데이터셋 생성 (스케일러 학습용)
train_dataset_temp = BoilingRegimeDataset(excel_file=train_label, root_dir=dataset_train)
train_features, train_targets = load_features_and_targets(train_dataset_temp)

# 스케일러 학습
feature_scaler.fit(train_features)
target_scaler.fit(train_targets)

# 데이터셋 생성 (학습된 스케일러 전달)
full_train_dataset = BoilingRegimeDataset(excel_file=train_label, root_dir=dataset_train,
                                          feature_scaler=feature_scaler, target_scaler=target_scaler)
test_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test,
                                    feature_scaler=feature_scaler, target_scaler=target_scaler)

train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# FNO 모델 정의 (변경 사항 없음)
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # 유지할 푸리에 모드 수

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # (batch, in_channels, x) * (in_channels, out_channels, modes) -> (batch, out_channels, x)
        return torch.einsum("bix, ioj -> box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, output_dim, num_layers=4):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        self.fc0 = nn.Linear(1, self.width)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(SpectralConv1d(self.width, self.width, self.modes))
            self.ws.append(nn.Conv1d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2_heat_flux = nn.Linear(128, 1)
        self.fc2_heat_transfer = nn.Linear(128, 1)
        self.fc2_boiling_regime = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # x: (batch, N, 1)
        x = self.fc0(x)          # x: (batch, N, width)
        x = x.permute(0, 2, 1)   # x: (batch, width, N)

        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = torch.relu(x)

        x = x.mean(-1)  # 글로벌 평균 풀링
        x = self.fc1(x)
        x = torch.relu(x)

        heat_flux = self.fc2_heat_flux(x)
        heat_transfer_coefficient = self.fc2_heat_transfer(x)
        boiling_regime = self.fc2_boiling_regime(x)

        return heat_flux, heat_transfer_coefficient, boiling_regime

class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, modes=16, width=32, output_dim=3, num_layers=4, learning_rate=0.001, num_epochs=300, batch_size=10, early_stopping_patience=50):
        self.modes = modes
        self.width = width
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

    def fit(self, X, y):
        self.model = FNO1d(self.modes, self.width, self.output_dim, num_layers=self.num_layers).to(device)
        criterion_classification = nn.CrossEntropyLoss()
        criterion_regression = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=max(2, self.batch_size), shuffle=True)

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
                loss_classification = criterion_classification(boiling_regime, labels[:, 2].long())
                loss_regression = criterion_regression(heat_flux.squeeze(), labels[:, 0]) + \
                                  criterion_regression(heat_transfer_coefficient.squeeze(), labels[:, 1])
                loss = loss_classification + loss_regression
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
                if early_stopping_counter >= self.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        self.model.load_state_dict(best_model_wts)
        self.classes_ = np.unique(y[:, 2])
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            data_loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=self.batch_size, shuffle=False)
            all_preds = []
            for features in data_loader:
                features = features.to(device)
                heat_flux, heat_transfer, boiling_regime = self.model(features)

                _, boiling_preds = torch.max(boiling_regime, 1)
                preds = torch.cat(
                    [heat_flux.cpu(), heat_transfer.cpu(), boiling_preds.float().cpu().unsqueeze(1)], dim=1)
                all_preds.extend(preds.numpy())
        return np.array(all_preds)

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

# 사용자 정의 스코어 함수 (변경 사항 없음)
def rmspe(y_true, y_pred):
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.max() - y_true.min())

def custom_score(y_true, y_pred):
    nrmse_heat_flux = nrmse(y_true[:, 0], y_pred[:, 0])
    nrmse_heat_transfer = nrmse(y_true[:, 1], y_pred[:, 1])
    accuracy_boiling = accuracy_score(y_true[:, 2].astype(int), y_pred[:, 2].astype(int))
    combined_score = (1 - nrmse_heat_flux) / 3 + (1 - nrmse_heat_transfer) / 3 + accuracy_boiling / 3
    return combined_score

# 데이터 준비
def prepare_data(dataset):
    features_list = []
    labels_list = []
    for features, heat_flux, heat_transfer_coefficient, boiling_regime in dataset:
        features_list.append(features.cpu().numpy())
        labels_list.append([heat_flux.cpu().item(), heat_transfer_coefficient.cpu().item(), boiling_regime.cpu().item()])
    X = np.array(features_list)
    y = np.array(labels_list)
    return X, y

X_train, y_train = prepare_data(full_train_dataset)
X_test, y_test = prepare_data(test_dataset)

custom_scorer = make_scorer(custom_score, greater_is_better=True)

num_classes = len(np.unique(y_train[:, 2]))
model = TorchModelWrapper(output_dim=num_classes)

# GridSearchCV 실행
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=custom_scorer, cv=fold, verbose=2, refit=True)
grid_search.fit(X_train, y_train)

# 결과 분석 및 저장 (변경 사항 없음)
def analyze_results(grid_search):
    results = []
    for params, mean_score, scores in zip(grid_search.cv_results_['params'],
                                          grid_search.cv_results_['mean_test_score'],
                                          grid_search.cv_results_['std_test_score']):
        modes = params['modes']
        width = params['width']
        num_layers = params['num_layers']
        results.append({
            'modes': modes,
            'width': width,
            'num_layers': num_layers,
            'mean_score': mean_score,
            'std_score': scores
        })
    return pd.DataFrame(results)

results_grid_df = analyze_results(grid_search)

best_model = grid_search.best_estimator_
torch.save(best_model.model.state_dict(), best_model_path)

with open(hyperparameters_path, 'w') as f:
    f.write(f'Best parameters: {grid_search.best_params_}\n')

# 테스트 (변경 사항 없음)
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
        features = torch.nan_to_num(features)
        heat_flux = heat_flux.to(device)
        heat_transfer_coefficient = heat_transfer_coefficient.to(device)
        boiling_regime = boiling_regime.to(device)
        outputs_heat_flux, outputs_heat_transfer_coefficient, outputs_boiling_regime = best_model.model(features)

        # 역변환
        heat_flux_np = heat_flux.cpu().numpy().reshape(-1, 1)
        heat_transfer_coefficient_np = heat_transfer_coefficient.cpu().numpy().reshape(-1, 1)
        outputs_heat_flux_np = outputs_heat_flux.cpu().numpy()
        outputs_heat_transfer_coefficient_np = outputs_heat_transfer_coefficient.cpu().numpy()

        heat_flux_true = target_scaler.inverse_transform(np.hstack([heat_flux_np, heat_transfer_coefficient_np]))[:, 0]
        heat_transfer_true = target_scaler.inverse_transform(np.hstack([heat_flux_np, heat_transfer_coefficient_np]))[:, 1]
        heat_flux_pred = target_scaler.inverse_transform(np.hstack([outputs_heat_flux_np, outputs_heat_transfer_coefficient_np]))[:, 0]
        heat_transfer_pred = target_scaler.inverse_transform(np.hstack([outputs_heat_flux_np, outputs_heat_transfer_coefficient_np]))[:, 1]

        _, predicted = torch.max(outputs_boiling_regime, 1)
        all_labels.extend(boiling_regime.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        all_heat_flux_true.extend(heat_flux_true)
        all_heat_flux_pred.extend(heat_flux_pred)
        all_heat_transfer_true.extend(heat_transfer_true)
        all_heat_transfer_pred.extend(heat_transfer_pred)

# 결과 저장 (변경 사항 없음)
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

# Confusion Matrix Plotting (변경 사항 없음)
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

print(f'Test results saved to {test_results_path}')
