import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTConfig
import seaborn as sns

# Load paths and settings
save_folder = './Results_Pool_100%_Spectrogram'

best_model_path = os.path.join(save_folder, 'transformer_best_model.pth')
output_results_path = os.path.join(save_folder, 'transformer_test_flowboiling-test_predictions.xlsx')
confusion_matrix_path = os.path.join(save_folder, 'transformer_confusion_matrix_flowboiling-test.png')

# Dataset paths for testing
dataset_test = './dataset_stft_flow_test'
test_label = './dataset_stft_flow_test/labels.xlsx'

batch_size = 50

# Image transform settings
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT input size
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
        self.target_scaler = target_scaler

    def __len__(self):
        return len(self.boiling_regime_frame)

    def __getitem__(self, idx):
        image_number = self.boiling_regime_frame.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, f'{image_number}')  # Assuming images are in .png format
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

        return image, torch.tensor([heat_flux, heat_transfer_coefficient]).float(), torch.tensor(boiling_regime).long()


class TransformerModel(torch.nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        config = ViTConfig(hidden_size=d_model, num_attention_heads=nhead, num_hidden_layers=num_layers)
        self.vit = ViTModel(config)
        self.fc_heat_flux = torch.nn.Linear(d_model, 1)
        self.fc_heat_transfer_coefficient = torch.nn.Linear(d_model, 1)
        self.fc_boiling_regime = torch.nn.Linear(d_model, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        x = outputs.last_hidden_state[:, 0]  # CLS token output
        heat_flux = self.fc_heat_flux(x).squeeze(-1)
        heat_transfer_coefficient = self.fc_heat_transfer_coefficient(x).squeeze(-1)
        boiling_regime = self.fc_boiling_regime(x)
        return heat_flux, heat_transfer_coefficient, boiling_regime


# Load scaler
train_scaler = MinMaxScaler()
full_train_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test, transform=transform)
train_scaler.fit(full_train_dataset.boiling_regime_frame.iloc[:, 2:4].values)

# Load test dataset and dataloader
test_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test, transform=transform, target_scaler=train_scaler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the best model
num_classes = len(test_dataset.boiling_regime_frame['BoilingRegime'].unique())

# Ensure the model configuration is the same as used in training
vit_config = ViTConfig(
    hidden_size=128,  # Use the same d_model as training
    num_attention_heads=8,
    num_hidden_layers=8
)
best_model = TransformerModel(num_classes=num_classes, d_model=128, nhead=8, num_layers=8)
best_model.vit = ViTModel(vit_config)

# Load state dict
best_model.load_state_dict(torch.load(best_model_path, map_location=device))
best_model.to(device)
best_model.eval()

# Testing loop
all_labels = []
all_preds = []
all_heat_flux_true = []
all_heat_flux_pred = []
all_heat_transfer_true = []
all_heat_transfer_pred = []

with torch.no_grad():
    for features, targets, boiling_regime in test_loader:
        features = features.to(device)
        targets = targets.to(device)

        outputs_heat_flux, outputs_heat_transfer_coefficient, outputs_boiling_regime = best_model(features)

        # Inverse transform
        outputs_heat_flux_np = outputs_heat_flux.cpu().numpy().reshape(-1, 1)
        outputs_heat_transfer_coefficient_np = outputs_heat_transfer_coefficient.cpu().numpy().reshape(-1, 1)
        true_values = targets.cpu().numpy()
        true_values_inverse = train_scaler.inverse_transform(true_values)
        pred_values = np.hstack([outputs_heat_flux_np, outputs_heat_transfer_coefficient_np])
        pred_values_inverse = train_scaler.inverse_transform(pred_values)

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

# Compute metrics
def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

nrmse_heat_flux = nrmse(all_heat_flux_true, all_heat_flux_pred)
nrmse_heat_transfer = nrmse(all_heat_transfer_true, all_heat_transfer_pred)

print(f'Test Heat Flux NRMSE: {nrmse_heat_flux}')
print(f'Test Heat Transfer Coefficient NRMSE: {nrmse_heat_transfer}')

# Save predictions and metrics
results_df = pd.DataFrame({
    'True Heat Flux': all_heat_flux_true,
    'Predicted Heat Flux': all_heat_flux_pred,
    'True Heat Transfer Coefficient': all_heat_transfer_true,
    'Predicted Heat Transfer Coefficient': all_heat_transfer_pred,
    'True Boiling Regime': all_labels,
    'Predicted Boiling Regime': all_preds
})

with pd.ExcelWriter(output_results_path) as writer:
    results_df.to_excel(writer, sheet_name='Predictions', index=False)

print(f'Test results saved to {output_results_path}')
