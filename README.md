# Pneumonia Detection from Chest X-Rays

A deep learning project for classifying chest X-ray images as **NORMAL** or **PNEUMONIA**, built with fastai, PyTorch, and a pretrained ResNet50 model in Google Colab.

> For educational use only. This is not a medical tool and must not be used to diagnose pneumonia or guide treatment.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uzmaa29/Pneumonia-Detection/blob/main/pnuemonia_prediction.ipynb)

## Overview

The notebook covers dataset download, image preprocessing, model training, and basic evaluation. It uses transfer learning to adapt a ResNet50 model pretrained on ImageNet to a two-class chest X-ray classification task.

**Next step:** a Flask web app is planned, where a user can upload a chest X-ray and see the model's prediction. It is not built yet.

### Current features

- Download the dataset using the Kaggle API.
- Load and resize images to 224 x 224 pixels with fastai.
- Train a ResNet50 model and track validation accuracy.
- Visualize predictions with a confusion matrix.
- Run predictions on individual X-ray images.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset shared by Paul Mooney on Kaggle.

The classification labels are:

- **NORMAL**
- **PNEUMONIA**

See the dataset page for its documentation, license, and terms of use.

## Method

| Step | Implementation |
| --- | --- |
| Data loading | `ImageDataLoaders.from_folder` |
| Image size | 224 x 224 pixels |
| Validation split | Random 20% split with `valid_pct=0.2` and `seed=67` |
| Model | Pretrained ResNet50 through `vision_learner` |
| Metric | Accuracy |
| Learning-rate exploration | `learn.lr_find()` |
| Training | `learn.fit_one_cycle(4, slice(1e-3, 1e-1))` |
| Evaluation | `ClassificationInterpretation` and a confusion matrix |
| Single-image inference | `learn.predict()` |

**Split caveat:** the notebook loads images from the combined dataset folders and creates a random validation split. It does not preserve the dataset's supplied train, validation, and test separation.

## Results

The recorded training run used a **Google Colab T4 GPU**.

The following values are rounded from the training output saved in the notebook:

| Epoch | Training loss | Validation loss | Validation accuracy |
| --- | --- | --- | --- |
| 0 | 0.5604 | 0.2742 | 94.75% |
| 1 | 0.2693 | 0.1127 | 97.05% |
| 2 | 0.1619 | 0.0997 | 96.46% |
| 3 | 0.0788 | 0.0682 | 97.57% |

Final recorded validation accuracy: **97.57%** on this random split.

The notebook also includes two manual predictions, one image labeled NORMAL and one labeled PNEUMONIA. Both predictions match their labels. These examples are demonstrations, not a separate test-set evaluation.

## Limitations

- Validation accuracy is not held-out test accuracy or evidence of clinical performance.
- Because the supplied dataset splits are pooled before splitting again, the manually selected images from the `test` folder are not guaranteed to be unseen during training.
- Only two manual predictions are shown. Precision, recall, and other metrics have not been reported on an independent test set.
- Performance on images from other hospitals, scanners, or patient populations has not been established.
- The current project is a notebook. A Flask app has not been implemented yet.

## Run the notebook

1. Open the notebook using the Colab badge above.
2. Select a GPU runtime for training if one is available.
3. Provide your own Kaggle API credentials in a `kaggle.json` file at `/content/kaggle.json`, as expected by the notebook. Do not commit this file to GitHub or share it.
4. Ensure the notebook's dependencies are installed: `fastai`, `torch`, `torchvision`, `pandas`, `numpy`, and the Kaggle CLI.
5. Run the cells in order to download and extract the dataset, train the model, and view the evaluation output.

The notebook uses Colab-specific paths under `/content`. Adjust those paths if running locally. Results may vary between runs and environments.

## Planned improvements

These changes are planned, not part of the current implementation:

- Refine and organize the project code.
- Build a Flask app for uploading an X-ray image and displaying the model's prediction.

## Tech stack

Python · fastai · PyTorch · ResNet50 · pandas · NumPy · Google Colab · Kaggle API

## Reference

- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), shared by Paul Mooney on Kaggle.
