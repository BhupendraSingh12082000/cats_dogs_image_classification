# Cat vs Dog Classifier Project

This project is a Deep learning-based image classification model that distinguishes between images of cats and dogs. It utilizes deep learning techniques and a convolutional neural network (CNN) to achieve high accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Introduction

The goal of this project is to build an image classification model that can identify whether a given image contains a cat or a dog. This is achieved using a convolutional neural network (CNN), a type of deep learning model particularly effective for image processing tasks.

## Dataset

The dataset used for this project is the popular [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data). It consists of labeled images of cats and dogs, divided into training and validation sets.

### Dataset Details:
- **Training Set**: Images used to train the model.
- **Validation Set**: Images used to validate model performance during training.

## Model Architecture

The classifier is built using a convolutional neural network (CNN). The architecture includes:

1. Convolutional Layers: Extract spatial features from the images.
2. Pooling Layers: Downsample feature maps to reduce dimensionality.
3. Fully Connected Layers: Perform classification.
4. Dropout Layers: Prevent overfitting by randomly disabling neurons during training.

### Summary of Model Layers:
- Input Layer: Image size `(150x150x3)`
- Convolutional Layers: 32, 64, 128 filters with ReLU activation
- Pooling: MaxPooling with a `(2x2)` filter
- Fully Connected Layers: Dense layers with `ReLU` and final `softmax` activation for binary classification

## Preprocessing

The images are preprocessed as follows:

1. Resized to 150x150 pixels.
2. Normalized pixel values to range [0, 1].
3. Data Augmentation applied:
   - Random rotations
   - Flipping
   - Zooming

## Training

The model is trained using the following configuration:

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Epochs**: 25

Training is performed on the training set, and performance is monitored on the validation set.

## Evaluation

The model's performance is evaluated using:

1. Accuracy: Proportion of correctly classified images.
2. Loss: Binary cross-entropy loss during training and validation.

Confusion matrix and precision-recall metrics are also used for detailed analysis.

## Dependencies

The following Python libraries are required:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/cat-vs-dog-classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cat-vs-dog-classifier
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Cat_vs_Dog_classifier.ipynb
   ```

## Results

The trained model achieves the following results:

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%

### Sample Predictions:

| Image          | Prediction  |
|----------------|-------------|
| Cat Image      | Cat         |
| Dog Image      | Dog         |

## Future Improvements

- Implement transfer learning with pre-trained models like VGG16 or ResNet.
- Optimize hyperparameters for better accuracy.
- Deploy the model as a web application using Flask or FastAPI.
- Enhance data augmentation techniques to improve generalization.

---

### Author

Developed by **Bhupendra Singh** as part of a hands-on Deep learning project.
