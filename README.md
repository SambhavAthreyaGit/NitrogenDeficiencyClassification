# Classifying Nitrogen Deficiency In Rice Crops

## Dataset
- **Source:** Kaggle — *Nitrogen Deficiency Image* dataset (`/NitrogenDeficiencyImage`).
- **Split (from run logs):** 5,390 training images and 400 test images across **4 classes**.
- **Task:** Multiclass image classification of nitrogen status from leaf/crop photos.

## What the Code Does (High Level)
- **Data pipeline:** Loads the Kaggle training and test folders; rescales pixels to [0,1]; applies on-the-fly augmentation (rotation, shifts, shear, zoom, horizontal flips).
- **Models:** Trains **two sequential CNNs** with increasing depth:
  - **Model 1:** 3× Conv2D blocks (32/64/64), MaxPooling, Flatten, Dense(32), Dropout(0.2), and a 4-way softmax head.
  - **Model 2:** 3× Conv2D blocks (32/64/128), MaxPooling, Flatten, Dense(64), Dropout(0.5), and a 4-way softmax head.
- **Training:** Categorical cross-entropy with Adam (lr=1e-3), early stopping on validation loss with best-weights restore; tracks accuracy/loss for train vs. validation; plots learning curves.

## Results (From This Run)
- **Model 1:** Best validation accuracy peaked around **~0.65** (e.g., 0.6475) before early stopping.
- **Model 2:** Best validation accuracy reached **~0.68** (e.g., 0.6750) with slightly deeper features and stronger dropout.
- **Observation:** Augmentation plus modest depth improved generalization; Model 2 consistently outperformed Model 1 on the held-out test fold used as validation during training.

## Notes
- Class names are dataset-defined; training used directory labels as provided by Kaggle.
- Curves suggest mild overfitting risk; additional regularization, more data, or transfer learning (e.g., a lightweight pretrained backbone) could further boost accuracy.

