# vision_sota_papers

This repository contains implementations and experiments of state-of-the-art computer vision models, starting with a modified AlexNet for the CIFAR-10 dataset, a reduced VGG16 for the Tiny ImageNet dataset, and now an implementation of GoogLeNet with auxiliary classifiers for Tiny ImageNet. It is intended for educational and research purposes, showcasing how classic and modern architectures perform on small- and medium-scale datasets.

---

## 📌 Modified AlexNet for CIFAR-10

This implementation adapts AlexNet to work efficiently with the CIFAR-10 dataset (32x32 images, 10 classes). The original AlexNet was designed for ImageNet-scale data, so this version modifies the input size, number of filters, and depth to suit smaller inputs.

### ⚙️ Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 128
- **Dropout Rate**: 0.5 (in classifier)
- **Activation Function**: ReLU
- **Epochs**: 40
- **Loss Function**: Cross-Entropy Loss

### 🏗 Architecture Overview
- Total of 6 learnable layers: 4 convolutional + 2 fully connected
- Input: 32x32x3 RGB image
- Includes Local Response Normalization (LRN) and MaxPooling after first two conv layers

### 📚 Dataset: CIFAR-10
- **Total Images**: 60,000 (50k train + 10k test)
- **Image Size**: 32×32×3
- **Number of Classes**: 10

### 📈 Results Summary
- **Final Test Accuracy**: **79.85%**
- **Final Test Loss**: **0.6039**

---

## 📌 Reduced VGG16 for Tiny ImageNet

This implementation modifies the original VGG16 architecture to reduce parameter count and make it more suitable for Tiny ImageNet (64x64 images, 200 classes). Channel counts are reduced in each layer while retaining the general depth and block structure.

### ⚙️ Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Batch Size**: 256
- **Dropout Rate**: 0.3 (in classifier)
- **Activation Function**: ReLU
- **Epochs**: 60
- **Loss Function**: Cross-Entropy Loss

### 🏗 Architecture Overview
- 13 convolutional layers + BatchNorm + ReLU
- 5 MaxPool layers
- Classifier with 2 fully connected layers (512 units each) + Dropout

### 📚 Dataset: Tiny ImageNet
- **Total Images**: 110,000 (100k train + 10k val)
- **Image Size**: 64×64×3
- **Number of Classes**: 200

### 📈 Results Summary
- **Final Val Accuracy**: ~42.3%
- **Top-5 Error**: ~31.6%

---

## 📌 GoogLeNet with Auxiliary Classifiers for Tiny ImageNet

This implementation follows the original GoogLeNet (Inception v1) architecture but is adapted for Tiny ImageNet’s smaller image size (64x64) and higher class count (200). It includes auxiliary classifiers during training to improve gradient flow and regularization.

### ⚙️ Hyperparameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 256
- **Weight Decay**: 1e-4
- **Dropout Rate**: 0.4 (main classifier), 0.7 (auxiliary classifiers)
- **Activation Function**: ReLU
- **Epochs**: 50
- **Loss Function**: Cross-Entropy Loss

### 🏗 Architecture Overview

#### Input:
- 64x64x3 RGB image

#### Layers:
1. **Conv-1**: 64 filters, 7×7, stride=2 → BN → ReLU → MaxPool
2. **Conv-2**: 192 filters, 3×3 → BN → ReLU → MaxPool
3. **Inception (3a)**: (64, (96,128), (16,32), 32)
4. **Inception (3b)**: (128, (128,192), (32,96), 64) → MaxPool
5. **Inception (4a)**: (192, (96,208), (16,48), 64) → **Auxiliary Classifier 1**
6. **Inception (4b)**: (160, (112,224), (24,64), 64)
7. **Inception (4c)**: (128, (128,256), (24,64), 64)
8. **Inception (4d)**: (112, (144,288), (32,64), 64) → **Auxiliary Classifier 2**
9. **Inception (4e)**: (256, (160,320), (32,128), 128) → MaxPool
10. **Inception (5a)**: (256, (160,320), (32,128), 128)
11. **Inception (5b)**: (384, (192,384), (48,128), 128)
12. Global AvgPool → Dropout → Fully Connected (1024 → 200)

#### Auxiliary Classifiers:
- Each takes the output from intermediate inception layers
- Adaptive AvgPool → 1×1 Conv → FC → Dropout → FC (to 200 classes)
- Used only during training to assist optimization

---

### 📚 Dataset: Tiny ImageNet
- **Total Images**: 110,000 (100k train + 10k val)
- **Image Size**: 64×64×3
- **Number of Classes**: 200

---

### 📈 Results Summary
| Epoch | Train Acc | Val Acc | Top-5 Error |
|-------|-----------|---------|-------------|
| 10    | 34.8%     | 28.4%   | 59.1%       |
| 20    | 51.2%     | 38.7%   | 36.9%       |
| 30    | 61.3%     | 41.2%   | 33.5%       |
| 40    | 67.0%     | 43.0%   | 31.9%       |
| 50    | 70.5%     | 44.2%   | 30.6%       |

**Observation:** Auxiliary classifiers improved gradient flow, resulting in faster convergence during early training. Validation accuracy reached ~44% top-1 and ~30% top-5 error, outperforming the reduced VGG16 on the same dataset.

---

## 📦 Upcoming Additions
This repo will be expanded with more SOTA vision models, including:
- ResNet variants
- Vision Transformers (ViT)
- EfficientNet
- Swin Transformers
- ConvNeXt

---

## 🧠 Citation & Credits
This repository is maintained as a hands-on exploration of deep learning architectures for computer vision.
Feel free to fork, clone, or contribute to the project.
