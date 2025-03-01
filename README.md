3D MRI-Based Alzheimer's Disease Classification Using Deep Learning
Overview
This project focuses on classifying Alzheimer's Disease (AD) and Cognitively Normal (CN) subjects using 3D MRI images. A deep learning approach is implemented with DenseNet121, a 3D convolutional neural network (CNN), to process NIfTI images and predict Alzheimer's presence.

Dataset & Preprocessing
1. Data Loading and Preprocessing
The dataset is loaded from CSV files (updated_train.csv and balanced_val.csv), containing MRI scan paths and labels.
Abbreviations "CN" (Cognitively Normal) and "AD" (Alzheimer's Disease) are mapped to full names for better readability.
Labels are encoded as 0 (Alzheimer’s Disease) and 1 (Cognitively Normal) for training.
2. MRI Image Processing
The NIfTI format (medical imaging standard) is loaded using NiBabel.
Each MRI scan is converted into a tensor and reshaped to (1, Depth, Height, Width) for compatibility with CNNs.
Preprocessing transformations include:
Intensity scaling to normalize pixel values.
Resizing each scan to a fixed 64×64×64 resolution.
Model Architecture: DenseNet121 (3D CNN)
The model is based on MONAI’s DenseNet121, a 3D CNN designed for volumetric medical image analysis.

Key Features:
Spatial Dimensions: 3D (to handle volumetric MRI scans).
Input Channels: 1 (since MRI scans are grayscale).
Output Channels: 2 (binary classification - Alzheimer’s vs. Normal).
Training Setup
1. Data Handling
Custom PyTorch Dataset (MRIDataset) is implemented for on-the-fly image loading & augmentation.
DataLoader efficiently loads batches with shuffling and pin_memory for better GPU performance.
2. Loss & Optimizer
Loss Function: CrossEntropyLoss() (since it’s a classification task).
Optimizer: AdamW with learning rate = 1e-4 (effective for weight decay).
Learning Rate Scheduler: ReduceLROnPlateau (reduces LR if validation loss stagnates).
3. Mixed Precision Training
Uses torch.amp.GradScaler() to optimize GPU memory usage and speed up training.
4. Gradient Accumulation
Instead of updating weights every batch, updates happen every 2 steps (helps when GPU memory is limited).
Validation & Early Stopping
A validation loop evaluates performance every epoch.
Early stopping prevents overfitting by stopping training if validation loss doesn’t improve for 5 epochs.
Sample Predictions are printed to visually inspect classification results.
Results & Model Saving
Best model (lowest validation loss) is saved as "best_model.pth".
Performance metrics include accuracy, loss tracking, and sample predictions.
Conclusion
This deep learning pipeline provides an efficient approach for Alzheimer's Disease classification using 3D MRI images. By leveraging DenseNet121 (3D CNN), mixed precision training, and gradient accumulation, the model ensures high accuracy while being computationally efficient.
