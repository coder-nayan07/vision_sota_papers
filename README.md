# vision_sota_papers

This repository contains implementations and experiments of state-of-the-art computer vision models, starting with a modified AlexNet for the CIFAR-10 dataset. It is intended for educational and research purposes, showcasing how classic and modern architectures perform on small-scale datasets.

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

## 📦 Upcoming Additions

This repo will be expanded with more SOTA vision models, including:

- ResNet variants
- Vision Transformers (ViT)
- EfficientNet
- Swin Transformers
- ConvNeXt

---

## 🧠 Citation & Credits

This repository is maintained as a hands-on exploration of deep learning architectures for computer vision. Feel free to fork, clone, or contribute to the project.

---
