# Pneumonia Detection from Chest X-Rays

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA**. It includes two model notebooks and a Flask web app that serves the Keras CNN.

> **For educational use only.** This is not a medical tool and must not be used to diagnose pneumonia or guide treatment.

[v]

## Overview

The project has three parts:

1. **Keras CNN notebook** (`pneumonia_keras.ipynb`): a CNN built from scratch with TensorFlow/Keras, trained and evaluated on a patient-separated split.
2. **Flask app** (`app.py`, `templates/index.html`): upload a chest X-ray and get a prediction from the saved Keras model.
3. **fastai ResNet50 notebook** (`pnuemonia_prediction.ipynb`): an earlier transfer-learning experiment, kept for reference.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Paul Mooney on Kaggle. See the dataset page for its license and terms of use.

Labels: **NORMAL** and **PNEUMONIA**.

## Method (Keras CNN)

| Step | Implementation |
| --- | --- |
| Image size | 224 x 224 RGB, scaled to 0 to 1 |
| Model | 3 Conv2D blocks (32, 64, 128 filters) with BatchNormalization and MaxPooling, then Dense(128), Dropout(0.5), and a sigmoid output |
| Augmentation | Small rotation, zoom, and horizontal flip (training only) |
| Class imbalance | Class weights computed from the training set |
| Split | Validation is taken from `train/` using a patient-grouped split, so no patient appears in both training and validation. The `test/` folder is used once, for final evaluation. |
| Training | Adam (learning rate 1e-3), binary cross-entropy, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint |

Patient IDs come from the filenames (`IM-xxxx` for NORMAL, `personxxx` for PNEUMONIA).

Split sizes: 4,163 training, 1,053 validation, and 624 test images.

## Results

Trained on Google Colab with a T4 GPU. Restored weights come from epoch 6, the best validation loss.

**Test set (624 images, never used for training or tuning):**

| Metric | Value |
| --- | --- |
| Accuracy | 86.5% |
| AUC | 0.925 |
| NORMAL recall | 88% (207 of 234) |
| PNEUMONIA recall | 85% (333 of 390) |

Confusion matrix (rows are true labels, columns are predictions):

| | Predicted NORMAL | Predicted PNEUMONIA |
| --- | --- | --- |
| **True NORMAL** | 207 | 27 |
| **True PNEUMONIA** | 57 | 333 |

**A note on earlier results:** an earlier fastai run reported 97.57% validation accuracy, but that split mixed images from the same patients into both training and validation, so it overstated performance. The Keras results above are the reliable numbers.

## Limitations

- Test accuracy is measured on one dataset, not on clinical data from other hospitals, scanners, or patient populations.
- Validation accuracy varied a lot between epochs during training, so results may change between runs.
- The model makes errors in both directions. Missed pneumonia cases (57 in the test set) are the most important type of error in a clinical setting.
- The Flask app is a demonstration. It has no authentication, no logging, and is meant to run locally.

## Run the Flask app

Requirements: Python 3.12 and a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000, upload a chest X-ray, and click **Predict**.

The app loads `pneumonia_cnn.keras`, which must be in the same folder as `app.py`.

## Train the model (Keras notebook)

1. Open `pneumonia_keras.ipynb` in Google Colab.
2. Select a GPU runtime (Runtime → Change runtime type → T4 GPU).
3. Add your Kaggle API key as `kaggle.json` in `/content/`. Do not commit this file to GitHub.
4. Run the cells in order. The notebook downloads the dataset, trains the model, evaluates it on the test set, and saves `pneumonia_cnn.keras`.

The notebook uses Colab paths under `/content`. Adjust these if running locally.

## Project structure

```
Pneumonia-Detection/
├── app.py                    # Flask app
├── templates/
│   └── index.html            # Upload page
├── pneumonia_cnn.keras       # Trained Keras CNN model
├── pneumonia_keras.ipynb     # Keras CNN training notebook
├── pnuemonia_prediction.ipynb # fastai ResNet50 notebook (reference)
├── requirements.txt
└── README.md
```

## Tech stack

Python · TensorFlow / Keras · Flask · NumPy · Pillow · pandas · scikit-learn · fastai (reference notebook) · Google Colab · Kaggle API

## Reference

- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney, Kaggle.
