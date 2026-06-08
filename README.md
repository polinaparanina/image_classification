# Image Classification: The Simpsons Characters

Multiclass classification (42 classes) using CNN and ResNet18 fine-tuning.  
Dataset: [Kaggle — journey-springfield](https://www.kaggle.com/competitions/journey-springfield). Metric: F1-score.

## Approach

**Baseline — custom CNN**
- 4 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool), 2 FC layers
- Input: 224×224, ImageNet normalization
- Stratified train/val split (75/25)

**Augmentation pipeline** (torchvision v2)
- Random horizontal flip, color jitter, rotation

**Transfer learning — ResNet18**
- Feature extractor mode (frozen backbone)
- Fine-tuning mode (full unfreezing)

**Class imbalance handling**
- `WeightedRandomSampler` — oversampling of rare characters
- `CrossEntropyLoss` with class weights

**Noise removal**
- Top-loss analysis: identified and removed mislabeled/corrupted images → improved F1

## Results

| Model | F1 (Kaggle) |
|---|---|
| Baseline CNN | see notebook |
| CNN + augmentation | see notebook |
| ResNet18 feature extractor | see notebook |
| ResNet18 fine-tuning | see notebook |
| ResNet18 fine-tuning + augmentation | see notebook |
| ResNet18 fine-tuning + augmentation + weight sampler | see notebook |
| ResNet18 fine-tuning + drop noisy data | see notebook |

## Repository structure

```
├── image_classification.ipynb   # full pipeline: data loading, training, evaluation
├── requirements.txt
└── data/                        # not tracked — download via gdown (links in notebook)
    ├── train/
    └── testset/
```

## Data

Downloaded automatically in the notebook via `gdown`. Run the first cells to fetch the data.

## Requirements

```bash
pip install -r requirements.txt
```
