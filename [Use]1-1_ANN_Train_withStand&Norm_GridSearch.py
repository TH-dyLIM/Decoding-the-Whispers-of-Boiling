import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, precision_score, recall_score, \
    f1_score
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# input - AE singal voltage data (61,441x2 time vs voltage)
# output - heat flux, heat transfer coefficient, boiling regime
# input - standization 해서 학습함.
# ouput value - normalization 해서 학습함.
# GridsearchCV 함수로 27개의 DNN 모델을 한번에 돌리고 최적 구조 찾는 코드.

# 저장 폴더 정의
save_folder = './Results_Pool_100%_AEsignal'
# 폴더가 없으면 생성
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 경로 설정
accuracy_path = os.path.join(save_folder, 'nn_optimized_accuracy_1.xlsx')
best_model_path = os.path.join(save_folder, 'nn_optimized_best_model_1.pth')
training_loss_path = os.path.join(save_folder, 'nn_optimized_training_loss_1.png')
confusion_matrix_path = os.path.join(save_folder, 'nn_optimized_confusion_matrix_1.png')
hyperparameters_path = os.path.join(save_folder, 'nn_optimized_hyperparameters_s1.txt')
test_results_path = os.path.join(save_folder, 'nn_optimized_test_results_1.xlsx')
# 추가적인 경로 설정
params_vs_score_path = os.path.join(save_folder, 'params_vs_score_1.png')
layers_vs_score_path = os.path.join(save_folder, 'layers_vs_score_1.png')

# 데이터셋 경로 설정
dataset_train = './dataset_signal_pool_train_100%'
dataset_test = './dataset_signal_pool_test_100%'
train_label = os.path.join(dataset_train, 'labels.xlsx')
test_label = os.path.join(dataset_test, 'labels.xlsx')

num_epochs = 500
batch_size = 50
learning_rate = 0.0001
early_stopping_patience = 50
fold = 4 # 이 코드에서는 k-fold 적용되고 2 이상 숫자 입력해야함.

# Define the parameter grid
param_grid = {
    'layer_sizes': [
       [10000, 5000, 5000, 2000, 500, 250]
    # [10,10]
    ],
    'learning_rate': [learning_rate],
    'batch_size': [batch_size],
    'num_epochs': [num_epochs],
    'early_stopping_patience': [early_stopping_patience],
    'batch_normalization': [True],
    'dropout_rate': [0.1]
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
        features = data['Voltage'].values.astype('float32').reshape(1, -1)

        if self.feature_scaler:
            features = self.feature_scaler.transform(features).flatten()
        features = torch.tensor(features).to(device)

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

# 모든 CSV 파일을 한 번에 로드
# 모든 CSV 파일을 한 번에 로드하여 GPU에 할당
def load_all_data(root_dir, boiling_regime_frame):
    features_list = []
    labels_list = []

    for i in range(len(boiling_regime_frame)):
        image_number = boiling_regime_frame.iloc[i, 0]
        csv_file = os.path.join(root_dir, f'{image_number}')
        data = pd.read_csv(csv_file)

        # GPU에 할당할 텐서로 변환
        features = torch.tensor(data['Voltage'].values.astype('float32').reshape(1, -1)).to(device)
        features_list.append(features)

        heat_flux = torch.tensor(boiling_regime_frame.iloc[i, 2].astype('float32')).to(device)
        heat_transfer_coefficient = torch.tensor(boiling_regime_frame.iloc[i, 3].astype('float32')).to(device)
        boiling_regime = torch.tensor(boiling_regime_frame.iloc[i, 4]).long().to(device)

        labels_list.append(torch.stack([heat_flux, heat_transfer_coefficient, boiling_regime]))

        print(f"Loaded {i+1}/{len(boiling_regime_frame)} files to GPU")

    features_array = torch.stack(features_list)
    labels_array = torch.stack(labels_list)

    return features_array, labels_array

# 스케일러 초기화 및 데이터셋 초기화
feature_scaler = StandardScaler()
target_scaler = MinMaxScaler()

full_train_dataset = BoilingRegimeDataset(excel_file=train_label, root_dir=dataset_train, feature_scaler=feature_scaler,
                                          target_scaler=target_scaler)
test_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test, feature_scaler=feature_scaler,
                                    target_scaler=target_scaler)

train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 데이터 로드
features_array, labels_array = load_all_data(dataset_train, full_train_dataset.boiling_regime_frame)
# 스케일러를 사용하여 데이터 스케일링 (GPU에서 직접 수행)
# features_array의 차원을 (batch_size, feature_size)로 변환
features_array = features_array.view(features_array.size(0), -1)
# 스케일러를 사용하여 데이터 스케일링 (GPU에서 직접 수행)
features_array = torch.tensor(feature_scaler.fit_transform(features_array.cpu().numpy())).to(device)
# 스케일링 후 다시 원래 차원으로 변환
features_array = features_array.view(features_array.size(0), 1, -1)
labels_array[:, :2] = torch.tensor(target_scaler.fit_transform(labels_array[:, :2].cpu().numpy())).to(device)

# NaN 값 확인 및 처리
features_array = torch.nan_to_num(features_array)
labels_array = torch.nan_to_num(labels_array)

# 스케일러를 사용하여 데이터 스케일링
#features_array = feature_scaler.fit_transform(features_array)
#labels_array[:, :2] = target_scaler.fit_transform(labels_array[:, :2])


class MultiOutputANNModel(nn.Module):
    def __init__(self, input_size, num_classes, layer_sizes, batch_normalization=False, dropout_rate=0.0):
        super(MultiOutputANNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        previous_size = input_size
        self.batch_normalization = batch_normalization

        for size in layer_sizes:
            self.layers.append(nn.Linear(previous_size, size))
            if self.batch_normalization:
                self.batch_norms.append(nn.BatchNorm1d(size))
            previous_size = size

        self.fc_heat_flux = nn.Linear(previous_size, 1)
        self.fc_heat_transfer_coefficient = nn.Linear(previous_size, 1)
        self.fc_boiling_regime = nn.Linear(previous_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_normalization and x.size(0) > 1:  # 배치 크기가 1보다 큰 경우에만 배치 정규화 적용
                x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)

        heat_flux = self.fc_heat_flux(x)
        heat_transfer_coefficient = self.fc_heat_transfer_coefficient(x)
        boiling_regime = self.fc_boiling_regime(x)

        return heat_flux, heat_transfer_coefficient, boiling_regime


class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, num_classes, layer_sizes=[10000, 5000, 1000], learning_rate=0.001, num_epochs=300, batch_size=10, early_stopping_patience=50, batch_normalization=False, dropout_rate=0.0):
        self.input_size = input_size
        self.num_classes = num_classes
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate

    def fit(self, X, y):
        self.model = MultiOutputANNModel(self.input_size, self.num_classes, self.layer_sizes, self.batch_normalization,
                                         self.dropout_rate).to(device)
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
                loss_regression = criterion_regression(heat_flux, labels[:, 0].view_as(heat_flux)) + \
                                  criterion_regression(heat_transfer_coefficient,
                                                       labels[:, 1].view_as(heat_transfer_coefficient))
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

                # Ensure heat_flux and heat_transfer have the same dimension as boiling_regime
                heat_flux = heat_flux.view(-1, 1)
                heat_transfer = heat_transfer.view(-1, 1)

                _, boiling_preds = torch.max(boiling_regime, 1)
                preds = torch.cat(
                    [heat_flux, heat_transfer, boiling_preds.float().view(-1, 1)], dim=1)
                all_preds.extend(preds.cpu().numpy())
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



from sklearn.metrics import make_scorer, accuracy_score


def analyze_results(grid_search):
    results = []
    for params, mean_score, scores in zip(grid_search.cv_results_['params'],
                                          grid_search.cv_results_['mean_test_score'],
                                          grid_search.cv_results_['std_test_score']):
        layer_sizes = params['layer_sizes']
        num_params = sum(layer_sizes)
        num_layers = len(layer_sizes)
        results.append({
            'layer_sizes': layer_sizes,
            'num_params': num_params,
            'num_layers': num_layers,
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

# Prepare the dataset for grid search
# Cross validation 하기 때문에 validation 데이터셋 필요 없음.
X_train = features_array
y_train = labels_array

# X_train과 y_train을 float32 타입의 텐서로 변환
X_train = X_train.to(torch.float32)
y_train = y_train.to(torch.float32)
y_train[:, 2] = y_train[:, 2].to(torch.int)  # Ensure boiling regime is integer

# Perform grid search
num_classes = len(torch.unique(labels_array[:, 2].cpu()))  # Ensure to use .cpu() for unique calculation
model = TorchModelWrapper(input_size=61441, num_classes=num_classes)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

custom_scorer = make_scorer(custom_score, greater_is_better=True)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=custom_scorer, cv=fold, verbose=2, refit=True)
grid_search.fit(X_train.cpu().numpy(), y_train.cpu().numpy())  # Use .cpu() and .numpy()

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
        features = features.to(device)
        features = torch.nan_to_num(features)
        heat_flux = heat_flux.to(device)
        heat_transfer_coefficient = heat_transfer_coefficient.to(device)
        boiling_regime = boiling_regime.to(device)
        outputs_heat_flux, outputs_heat_transfer_coefficient, outputs_boiling_regime = best_model.model(features)

        # Inverse transform
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