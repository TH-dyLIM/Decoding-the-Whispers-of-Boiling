# A Deep Learning Framework with Acoustic Signatures for Boiling Monitoring

## About This Repository
Welcome to the repository for our research on "Decoding the Whispers of Boiling: A Deep Learning Framework with Acoustic Signatures for Boiling Monitoring".
- **Doyeong Lim**
- **Yang Liu**
- **In Cheol Bang** (Corresponding Author)

**Email (Doyeong Lim):** [dylim@tamu.edu]

## Research Highlights
- **Nonintrusive acoustic-based monitoring:** Our approach uses external acoustic signals, eliminating the need for direct visualization or intrusive sensors. This enables accurate boiling diagnostics even in harsh, nontransparent, and radiation-intensive environments.
- **Advanced deep learning frameworks:** By employing state-of-the-art models—including transformers and Fourier neural operators—on spectrogram-derived features, we achieve high predictive accuracy and robustness against noise, ensuring reliable inference of heat flux, heat transfer coefficient, and boiling regime.
- **Broad applicability and industrial impact:** The proposed method not only delivers ±20% accuracy for critical boiling parameters and ~98% classification accuracy but also generalizes well to flow boiling scenarios. Its successful application across varying conditions can enhance safety margins, operational efficiency, and real-time control in diverse thermal energy systems.

## Contact & Collaboration
For inquiries, collaborations, or feedback, please reach out to the first author, Doyeong Lim, at [dylim@tamu.edu](mailto:dylim@tamu.edu).


---
## TF-Spec Model for Boiling AE Analysis
This repository provides an example of how to train and evaluate a Transformer-based model (TF-Spec) for predicting boiling heat flux, heat transfer coefficient (HTC), and boiling regime classification from Acoustic Emission (AE) spectrogram images. The code demonstrates a grid search approach over specified hyperparameters, as well as various training and evaluation steps including data loading, inference, performance metrics, and result visualization.

## Table of Contents
1. [Prerequisites](#1-prerequisites)  
2. [Project Structure](#2-project-structure)  
3. [Data Preparation](#3-data-preparation)  
4. [Usage Instructions](#4-usage-instructions)  
5. [Training and Evaluation](#5-training-and-evaluation)  
6. [Important Notes](#6-important-notes)  
7. [Future Extensions](#7-future-extensions)  

---

## 1. Prerequisites
- **Python** 3.7+
- **PyTorch** 1.10+
- **torchvision** 0.11+
- **transformers** (for Vision Transformer support)
- Other libraries: **NumPy**, **Pandas**, **scikit-learn**, **Matplotlib**, **Seaborn**, **openpyxl**

Install the core dependencies via:
```bash
pip install torch torchvision transformers pandas scikit-learn matplotlib seaborn openpyxl

## 2. Project Structure
A typical structure for your repository or local folder could look like:

├── dataset_stft_pool_train_100%/
│   ├── 00001.png
│   ├── 00002.png
│   ├── ...
│   └── labels.xlsx
├── dataset_stft_pool_test_100%/
│   ├── 01000.png
│   ├── 01001.png
│   ├── ...
│   └── labels.xlsx
├── tf_spec_model.py
├── Results_Pool_100%_Spectrogram/
│   ├── transformer_accuracy.xlsx
│   ├── transformer_best_model.pth
│   ├── transformer_confusion_matrix.png
│   ├── transformer_hyperparameters.txt
│   ├── transformer_params_vs_score.png
│   ├── transformer_test_results.xlsx
│   └── transformer_training_loss.png
└── README.md


dataset_stft_pool_train_100%/ and dataset_stft_pool_test_100%/: Folders containing your training and test spectrogram images, along with an labels.xlsx file that links each image to its true heat flux, HTC, and boiling regime.

tf_spec_model.py: Example Python script illustrating the entire training pipeline, including data loading, model definition (ViT-based), training, grid search, and evaluation.

Results_Pool_100%_Spectrogram/: This folder is automatically created (if not present) to store training outcomes, best model weights, confusion matrices, metric plots, etc.

## 3. Data Preparation
Spectrogram Images

Convert raw AE signals into spectrogram images (e.g., 224×224 pixels). Each image should reflect the time-frequency representation of the signal.

Store these images in a consistent naming format (e.g., 00001.png, 00002.png, ...).

Labels File (labels.xlsx)

The labels.xlsx file should map each image name (without file extension) to the corresponding heat flux, HTC, and boiling regime index (e.g., 0 for natural convection, 1 for nucleate boiling, 2 for CHF).

The code assumes columns in this order: [image_number, ..., heat_flux, HTC, boiling_regime, ...].

Make sure the column indices or names match how they are accessed in the script.

Folder Structure

Place training images under a folder like dataset_stft_pool_train_100%/ alongside a labels.xlsx.

Do the same for your test dataset under dataset_stft_pool_test_100%/.


## 4. Usage Instructions
Clone or Download the Repository

git clone https://github.com/YourUser/YourRepo.git (adapt if needed)

Adjust Paths

Inside tf_spec_model.py, you will see variables such as:

python
dataset_train = './dataset_stft_pool_train_100%'
dataset_test = './dataset_stft_pool_test_100%'
train_label = './dataset_stft_pool_train_100%/labels.xlsx'
test_label = './dataset_stft_pool_test_100%/labels.xlsx'
save_folder = './Results_Pool_100%_Spectrogram'
Modify these paths to point to the correct directories on your system.

Run the Script

From a terminal or IDE, execute:

python tf_spec_model.py
The code will perform a grid search over the specified hyperparameters (param_grid), then train the best model configuration, and finally evaluate it on the test set.

Monitor Outputs

Look for output logs detailing each training epoch, plus final metrics (MSE, NRMSE, RMSPE, confusion matrix, etc.) on the test set.



## 5. Training and Evaluation
Grid Search
The code uses GridSearchCV from scikit-learn to iterate over hyperparameter combinations (e.g., number of transformer layers, learning rate, batch size, etc.).

Early Stopping
Early stopping is implemented based on a patience counter (default 50 epochs), which halts training if validation loss fails to improve.

Metrics
We calculate various regression and classification metrics, including MSE, NRMSE, RMSPE, MAPE, confusion matrix, and accuracy.

Key Model Components:

BoilingRegimeDataset: Custom torch.utils.data.Dataset that loads images and corresponding labels from Excel.

TransformerModel: A Vision Transformer-based network (ViT) built with transformers.ViTModel.

TorchModelWrapper: A scikit-learn compatible wrapper, enabling usage of GridSearchCV.

custom_score(): A combined regression–classification metric function used to evaluate performance during grid search.

## 6. Important Notes
Path Adjustments
You must adapt file and folder paths to match your environment. For instance, the code references local folders like ./dataset_stft_pool_train_100%/; if your data is stored elsewhere, update these paths accordingly.

Data Splits

By default, this script assumes that you have separate training and test folders. If you need cross-validation, ensure your folder structure and label files are aligned with such a setup.

Hardware Requirements
Running the Vision Transformer can be GPU-intensive. For larger datasets or more layers, a high-memory GPU (Best optimized RTX 3090) is recommended.

Hyperparameter Tuning
The provided param_grid can be customized to explore other learning rates, batch sizes, or model configurations. The script demonstrates one potential search space.

Multiple Classes
For boiling regime classification, ensure your label indices (e.g., 0, 1, 2) and confusion matrix logic match your dataset categories.

7. Future Extensions
Additional Architectures
This example focuses on a Vision Transformer (ViT). You can easily modify it to test ResNet, EfficientNet, or other CNN-based backbones.

Transfer Learning
Adapt the same pipeline for fine-tuning across different boiling conditions or fluid regimes, using a portion of data for re-training.

Model Interpretability
Libraries like SHAP or Grad-CAM can be integrated to visualize how the model interprets specific frequency/time regions of the spectrogram.

