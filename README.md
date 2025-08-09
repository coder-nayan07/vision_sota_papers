# vision_sota_papers

This repository contains implementations and experiments of state-of-the-art computer vision models, starting with a modified AlexNet for the CIFAR-10 dataset and now extending to a reduced VGG16 for the Tiny ImageNet dataset. It is intended for educational and research purposes, showcasing how classic and modern architectures perform on small- and medium-scale datasets.

---

## 📌 Modified AlexNet for CIFAR-10

This implementation adapts AlexNet to work efficiently with the CIFAR-10 dataset (32x32 images, 10 classes). The original AlexNet was designed for ImageNet-scale data, so this version modifies the input size, number of filters, and depth to suit smaller inputs.

---

### ⚙️ Hyperparameters

- **Optimizer**: Adam  
- **Learning Rate**: 0.001  
- **Batch Size**: 128  
- **Dropout Rate**: 0.5 (in classifier)  
- **Activation Function**: ReLU  
- **Epochs**: 40  
- **Loss Function**: Cross-Entropy Loss

---

### 🏗 Architecture Overview

Total of 6 learnable layers: 4 convolutional + 2 fully connected.

#### Input:
- 32x32x3 RGB image

#### Layers:

1. **Conv-1**: 64 filters, 5×5, stride=1, padding=2  
   → ReLU → Local Response Normalization (LRN) → MaxPool (3×3, stride=2, padding=1)  
   → Output: 16×16×64

2. **Conv-2**: 64 filters, 5×5, stride=1, padding=2  
   → ReLU → LRN → MaxPool (3×3, stride=2, padding=1)  
   → Output: 8×8×64

3. **Conv-3**: 64 filters, 3×3, stride=1, padding=1  
   → ReLU  
   → Output: 8×8×64

4. **Conv-4**: 32 filters, 3×3, stride=1, padding=1  
   → ReLU  
   → Output: 8×8×32

5. **Flatten**: 2048 units (8×8×32)

6. **Fully Connected Layer**: 2048 → 10 (class logits)  
   → Dropout(p=0.5)

---

### 📚 Dataset: CIFAR-10

- **Total Images**: 60,000 (50k train + 10k test)  
- **Image Size**: 32×32×3  
- **Number of Classes**: 10  
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

### 📈 Results Summary

- **Final Test Accuracy**: **79.85%**
- **Final Test Loss**: **0.6039**

Training and validation accuracy/loss were logged for all 40 epochs. Below is a snapshot:

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1     | 36.03%    | 48.51%  | 1.7279     | 1.4316   |
| 10    | 70.07%    | 70.96%  | 0.8465     | 0.8293   |
| 20    | 76.13%    | 75.44%  | 0.6892     | 0.7250   |
| 30    | 78.42%    | 77.61%  | 0.6221     | 0.6596   |
| 40    | 80.13%    | 80.03%  | 0.5751     | 0.6091   |

> Full logs are available in the `logs/` directory.

---

## 📌 Reduced VGG16 for Tiny ImageNet

This implementation modifies the original VGG16 architecture to reduce parameter count and make it more suitable for Tiny ImageNet (64x64 images, 200 classes). Channel counts are reduced in each layer while retaining the general depth and block structure.

---

### ⚙️ Hyperparameters

- **Optimizer**: Adam  
- **Learning Rate**: 0.0001  
- **Batch Size**: 256  
- **Dropout Rate**: 0.3 (in classifier)  
- **Activation Function**: ReLU  
- **Epochs**: 60  
- **Loss Function**: Cross-Entropy Loss

---

### 🏗 Architecture Overview

#### Input:
- 64x64x3 RGB image

#### Feature Extractor:
- 13 convolutional layers with BatchNorm + ReLU activations  
- 5 MaxPool layers (2×2, stride=2)  
- Output volume after final pooling: 2×2×256

#### Classifier:
1. Flatten → 1024 features  
2. FC-1: 512 units → ReLU → Dropout(p=0.3)  
3. FC-2: 512 units → ReLU → Dropout(p=0.3)  
4. FC-3: 200 units (class logits)

---

### 📚 Dataset: Tiny ImageNet

- **Total Images**: 110,000 (100k train + 10k val)  
- **Image Size**: 64×64×3  
- **Number of Classes**: 200

---

### 📈 Results Summary

From epoch 54 to 60:

| Epoch | Train Acc | Val Acc | Val Top-5 Error |
|-------|-----------|---------|-----------------|
| 54    | 50.31%    | 41.09%  | 32.25%          |
| 56    | 51.23%    | 41.92%  | 32.01%          |
| 58    | 51.72%    | 41.81%  | 31.80%          |
| 60    | 52.85%    | 42.32%  | 31.66%          |

**Observation:** Training accuracy steadily increased over the final 7 epochs, with validation accuracy improving modestly, indicating controlled overfitting and stable convergence.

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
# vision_sota_papers
