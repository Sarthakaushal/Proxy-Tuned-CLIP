# Proxy-Tuned-CLIP

A PyTorch implementation of Proxy Tuning for CLIP (Contrastive Language-Image Pre-training) models, featuring weighted proxy tuning with different alpha values.

## Overview

This project implements proxy tuning for CLIP models to improve their performance on image classification tasks. The implementation includes:
- Expert CLIP model (ResNet18 + DistilBERT)
- Anti-Expert CLIP model (ResNet18 + DistilBERT) 
- Base Target model (ResNet50 + DistilBERT)
- Proxy-tuned Target model

## Dataset

The project uses the EuroSAT dataset, which contains satellite images taken from the Sentinel-2 satellite. The dataset addresses land use and land cover classification challenges using freely accessible Sentinel-2 satellite images from the Copernicus Earth observation program.

### Classes
The dataset includes 10 different land use and land cover classes:
1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### Image Characteristics
- Format: RGB images
- Resolution: Fixed size (standardized to match model input requirements)
- Properties:
  - Most classes show uniform distribution of pixel intensities
  - Classes like Forest, Pasture, and SeaLake tend to have darker pixel intensities
  - Industrial and Residential areas show high contrast
  - All images are professionally annotated and quality-controlled

## Results

| Model Name | Trained | Classification Accuracy |
|------------|---------|------------------------|
| Anti-Expert CLIP small | ❌ | 7% |
| Expert CLIP small | ✅ | 97% |
| Base Target CLIP | ❌ | 20% |
| Tuned Target CLIP | ✅ | 98% |

The proxy tuning approach successfully improved the base large model's accuracy by 49% after tuning.

## Installation
## Steps to start training and Proxy Training
1. Clone the repo
````bash
git clone <repo-url>
````
2. cd into the dir
````bash
cd Proxy-Tuned-Clip
````
3. Get dataset
````bash
kaggle datasets download -d apollo2506 eurosat-dataset
````
4. Run the following scripts
````bash
pip install -r requirements.txt

python3 train_expert.py
python3 train_target.py
python3 inference.py
````

## Model Architecture

The implementation uses:
- Image Encoders: 
  - ResNet18 (Expert & Anti-Expert models)
  - ResNet50 (Target model)
- Text Encoder: DistilBERT
- Projection heads for both image and text embeddings