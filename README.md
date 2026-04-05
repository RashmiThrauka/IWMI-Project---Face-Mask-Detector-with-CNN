# Face Mask Detection with Custom CNN
### IWMI Data Science Internship Assessment
A custom-built Convolutional Neural Network (CNN) that classifies face images as **masked** or **unmasked**, deployed as an interactive web application using Streamlit.

## Results
| Metric | Score |
|--------|-------|
| Training Accuracy | 96.75% |
| Validation Accuracy | 94.26% |
| Test Accuracy | 96% |
| Precision (with_mask) | 98% |
| Recall (with_mask) | 94% |
| F1-Score | 96% |

## Model Architecture
A custom CNN built from scratch using TensorFlow/Keras:
- 3x Convolutional blocks (Conv2D + BatchNormalization + MaxPooling2D)
- Filters: 32 → 64 → 128
- Flatten → Dense(128) → Dropout(0.5) → Dense(2, Softmax)
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Learning Rate Scheduler: ReduceLROnPlateau

## Dataset
Download the dataset from the link below and place it in your working directory:

[Dataset Download Link](https://drive.google.com/file/d/1Dw0DGHwdmiblqzk8u1LeMzMCqo87sJhN/view?usp=sharing)

Two classes:
- `with_mask` — 3725 images
- `without_mask` — 3828 images

## Project Structure
```
IWMI-Project---Face-Mask-Detector-with-CNN/
├── IWMI.ipynb              # Full training pipeline
├── app.py                  # Streamlit web application
├── best_model.h5           # Saved model weights
├── Result/
│   ├── training_curves.png
│   └── confusion_matrix.png
└── README.md
```
## How the app Works
1. Upload a `.jpg`, `.jpeg`, or `.png` face image
2. Haar Cascade detects the face location in the image
3. Detected face is cropped, resized to 128x128, and normalized
4. Custom CNN classifies as `with_mask` or `without_mask`
5. Result shown with confidence percentage and prediction bar chart
