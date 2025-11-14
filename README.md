# ❄️ U-Net Replicating

This repository contains a PyTorch replication of the **U-Net: Convolutional Networks for Biomedical Image Segmentation** model. The goal is to reproduce the **U-Net architecture** for accurate pixel-wise biomedical image segmentation.

> ⚠️ Note: Elastic deformation is **not included** in the augmentation. Only rotation and shift are implemented.

- Only the **original U-Net configuration** is fully implemented.  
- The network consists of a **contracting path (encoder)**, a **bottleneck**, and an **expansive path (decoder)** with skip connections (mirroring) to preserve spatial details.  
- The implementation uses 3×3 convolutions, ReLU activations, max pooling, and 2×2 up-convolutions.  

**Paper:** [U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)](https://arxiv.org/abs/1505.04597)

---


## 🎺 Model Architecture Overview

The U‑Net model consists of **four main parts**: contracting path, bottleneck, expansive path, and final output layer.

### 1️⃣ Contracting Path (Encoder)  
- Reduces spatial dimensions while increasing feature channels.  
- Each step: **2×3×3 Conv + ReLU → MaxPool (2×2)**.  

```python
s1 = self.enc1(x)              # 512x512 -> 512x512, 1->64
s2 = self.enc2(self.pool(s1))  # 512x512 -> 256x256, 64->128
s3 = self.enc3(self.pool(s2))  # 256x256 -> 128x128, 128->256
s4 = self.enc4(self.pool(s3))  # 128x128 -> 64x64, 256->512
```

### 2️⃣ Bottleneck
- The deepest layer with smallest spatial size but highest feature channels.
- Compresses information before decoder:
```python
b = self.bottleneck(self.pool(s4))  # 64x64 -> 32x32, 512->1024
```
### 3️⃣ Expansive Path (Decoder)

- Upsamples feature maps, concatenates corresponding encoder features (skip connection / mirroring), then applies ConvBlock.

```python
d4 = self.up4(b, s4)  # 32x32 -> 64x64
d3 = self.up3(d4, s3) # 64x64 -> 128x128
d2 = self.up2(d3, s2) # 128x128 -> 256x256
d1 = self.up1(d2, s1) # 256x256 -> 512x512
```
- This recovers spatial details lost during encoding and sharpens object boundaries.
 
### 4️⃣ Final Layer

- 1×1 Conv reduces feature channels to the number of classes.
```python
out = self.final(d1)  # Pixel-wise class prediction
```
---
## Project Structure
```bash
U-Net-Replicating/
│
├── models/
│   ├── unet.py                # Contracting + Expanding path, full U-Net architecture
│   ├── conv_block.py          # 3x3 Conv + ReLU + 3x3 Conv + ReLU block
│   ├── upconv_block.py        # Up-conv (2x2) + concat + conv block
│   └── init_weights.py        # He initialization function
│
├── training_utils/
│   ├── augmentation.py        # Elastic deformation, rotation, shift, gray variation
│   ├── loss.py                # Pixel-wise softmax + weighted cross-entropy
│   └── optimizer.py           # Example: SGD + momentum function
│
├── configs/
│   └── config.py              # Example parameters: batch size=1, momentum=0.99, tile size, lr
│
├── scripts/
│   └── preprocess_dataset.py  # Tile cropping, augmentation showcase
│
├── requirements.txt           # Python dependencies
└── README.md                  # Paper summary, usage, references
```

---

## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)





