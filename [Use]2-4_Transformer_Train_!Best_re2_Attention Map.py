import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from transformers import ViTConfig
from torch.utils.data import DataLoader, Dataset
from transformers.models.vit.modeling_vit import ViTSelfAttention

# Load paths and settings
save_folder = './Results_Pool_100%_Spectrogram'
save_folder_map = './Results_Pool_100%_Spectrogram/AttentionMAP'

best_model_path = os.path.join(save_folder, 'transformer_best_model.pth')
output_results_path = os.path.join(save_folder, 'transformer_test_flowboiling-test_predictions.xlsx')
confusion_matrix_path = os.path.join(save_folder, 'transformer_confusion_matrix_flowboiling-test.png')

# Dataset paths for testing
dataset_test = './dataset_stft_pool_test_100%'
test_label = './dataset_stft_pool_test_100%/labels.xlsx'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
   
if not os.path.exists(save_folder_map):
    os.makedirs(save_folder_map)

batch_size = 1  # Use batch size of 1 for per-image processing
num_classes = 3  # 실제 클래스 수로 설정

# Transformer model parameters
input_dim = 3  # RGB spectrogram
d_model = 128
nhead = 8
num_layers = 8

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset class
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

# Modify ViT Self-Attention module to store gradients
class ViTSelfAttentionGradCAM(ViTSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, grad):
        self.attn_gradients = grad

    def get_attn_gradients(self):
        return self.attn_gradients

    def get_attention_map(self):
        return self.attention_map

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # Original forward implementation
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        if output_attentions:
            self.attention_map = attention_probs
            attention_probs.register_hook(self.save_attn_gradients)

        # Apply dropout and head mask if needed (omitted for brevity)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        if output_attentions:
            outputs += (attention_probs,)
        return outputs

# Replace ViT modules to use the modified self-attention
from transformers.models.vit.modeling_vit import ViTModel, ViTEncoder, ViTLayer, ViTAttention, ViTSelfOutput

class ViTAttentionGradCAM(ViTAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = ViTSelfAttentionGradCAM(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # Add attentions if we output them
        return outputs

class ViTLayerGradCAM(ViTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = ViTAttentionGradCAM(config)
        # MLP and LayerNorm remain the same

class ViTEncoderGradCAM(ViTEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([ViTLayerGradCAM(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

class ViTModelGradCAM(ViTModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ViTEncoderGradCAM(config)

# Define the Transformer model using the modified ViTModel
class TransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        config = ViTConfig(
            hidden_size=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            output_attentions=True,
            hidden_act='relu',
        )
        self.vit = ViTModelGradCAM(config)
        self.fc_heat_flux = nn.Linear(d_model, 1)
        self.fc_heat_transfer_coefficient = nn.Linear(d_model, 1)
        self.fc_boiling_regime = nn.Linear(d_model, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        x = outputs.last_hidden_state[:, 0]  # [CLS] token
        heat_flux = self.fc_heat_flux(x).squeeze()
        heat_transfer_coefficient = self.fc_heat_transfer_coefficient(x).squeeze()
        boiling_regime = self.fc_boiling_regime(x)
        attentions = outputs.attentions  # List of attention maps
        return heat_flux, heat_transfer_coefficient, boiling_regime, attentions

# Function to generate attention maps
def generate_attention_map(model, input_tensor, target_index=None, task='classification'):
    """
    Generates the Grad-CAM attention map for a specific task.

    :param model: Trained TransformerModel
    :param input_tensor: Input image tensor, shape: [1, 3, 224, 224]
    :param target_index: For classification, the target class index
    :param task: 'heat_flux', 'heat_transfer_coefficient', or 'classification'
    :return: Attention map, shape: [num_patches]
    """
    model.eval()
    input_tensor.requires_grad = True

    # Forward pass
    heat_flux, heat_transfer_coefficient, boiling_regime, _ = model(input_tensor)
    if task == 'heat_flux':
        output = heat_flux
    elif task == 'heat_transfer_coefficient':
        output = heat_transfer_coefficient
    elif task == 'classification':
        if target_index is None:
            target_index = boiling_regime.argmax().item()
        output = boiling_regime[:, target_index]
    else:
        raise ValueError("Invalid task. Choose from 'heat_flux', 'heat_transfer_coefficient', 'classification'.")

    # Backward pass
    model.zero_grad()
    output.backward(retain_graph=True)

    # Get gradients and attention maps from the last ViT layer
    grads = model.vit.encoder.layer[-1].attention.attention.get_attn_gradients()
    cams = model.vit.encoder.layer[-1].attention.attention.get_attention_map()

    # Grad-CAM calculation
    grad = grads.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
    cam = cams.mean(dim=1).squeeze(0)    # [seq_len, seq_len]

    # Element-wise multiplication
    weights = grad  # [seq_len, seq_len]
    cam = (cam * weights).mean(dim=0)    # [seq_len]

    # Exclude CLS token and reshape
    cam = cam[1:]  # [seq_len - 1]
    num_patches = int(np.sqrt(cam.size(0)))
    cam = cam.reshape(num_patches, num_patches)

    # Normalize
    cam = cam.cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

# Function to visualize and save the attention map without titles and without white borders
def visualize_cam(input_image, cam, save_path=None):
    """
    Visualizes and saves the attention map overlaid on the input image without titles and white borders.

    :param input_image: PIL Image
    :param cam: Attention map, shape: [224, 224]
    :param save_path: Path to save the image
    """
    # Resize input image
    input_image_resized = input_image.resize((224, 224))
    image_np = np.array(input_image_resized)

    # Upsample CAM to image size
    cam = torch.tensor(cam)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Create a figure with no padding and no borders
    fig, ax = plt.subplots(figsize=(224/100, 224/100), dpi=200)  # figsize in inches, DPI changed to 200
    ax.imshow(image_np)
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.axis('off')  # Remove axes

    # Adjust layout to remove any padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure without any white borders
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)  # DPI changed to 200
    plt.close(fig)  # Close the figure to save memory

# Prepare the test dataset
test_dataset = BoilingRegimeDataset(excel_file=test_label, root_dir=dataset_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the saved best model

model = TransformerModel(num_classes=num_classes, d_model=d_model, nhead=nhead, num_layers=num_layers)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)

# Iterate over the test dataset and generate attention maps
for idx, (image_tensor, heat_flux, heat_transfer_coefficient, boiling_regime) in enumerate(test_loader):
    # Get the input image file name without extension
    sample_image_name = test_dataset.boiling_regime_frame.iloc[idx, 0]
    sample_image_full_path = os.path.join(dataset_test, f'{sample_image_name}')
    sample_image = Image.open(sample_image_full_path).convert('RGB')

    # Extract base file name without extension
    base_filename = os.path.splitext(os.path.basename(sample_image_name))[0]

    image_tensor = image_tensor.to(device)

    # Generate attention maps for each task
    with torch.enable_grad():
        # Heat Flux
        cam_heat_flux = generate_attention_map(model, image_tensor, task='heat_flux')
        # Heat Transfer Coefficient
        cam_heat_transfer = generate_attention_map(model, image_tensor, task='heat_transfer_coefficient')
        # Classification
        _, _, boiling_regime_output, _ = model(image_tensor)
        predicted_class = boiling_regime_output.argmax().item()
        cam_classification = generate_attention_map(model, image_tensor, target_index=predicted_class, task='classification')

    # Save the attention maps using the base file name
    save_path_heat_flux = os.path.join(save_folder_map, f'{base_filename}_heat_flux.png')
    save_path_heat_transfer = os.path.join(save_folder_map, f'{base_filename}_heat_transfer.png')
    save_path_classification = os.path.join(save_folder_map, f'{base_filename}_classification.png')

    # Removed the title parameter to ensure no titles are added
    visualize_cam(sample_image, cam_heat_flux, save_path=save_path_heat_flux)
    visualize_cam(sample_image, cam_heat_transfer, save_path=save_path_heat_transfer)
    visualize_cam(sample_image, cam_classification, save_path=save_path_classification)

    print(f'Processed image {idx+1}/{len(test_loader)}')

print('Attention maps have been generated and saved.')
