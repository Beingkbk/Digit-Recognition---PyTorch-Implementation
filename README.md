# MNIST Classification Using CNN in PyTorch

This repository provides a PyTorch implementation for classifying handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The code includes steps for dataset loading, model definition, training, and evaluation.

---

## Dataset

The MNIST dataset contains grayscale images of handwritten digits (0-9):
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images

The dataset is loaded using `torchvision.datasets.MNIST` and normalized to a range of [0, 1] using the `ToTensor()` transformation.

---

## Data Preparation

We use PyTorch's `DataLoader` to handle data batching:
- **Batch Size:** 100
- **Shuffling:** Enabled for the training set to improve learning.

---

## Model Architecture

The Convolutional Neural Network (CNN) consists of:
1. **Convolutional Layers:** Extract features from images.
2. **Pooling Layers:** Downsample feature maps.
3. **Dropout Layers:** Prevent overfitting.
4. **Fully Connected Layers:** Map features to the output classes.
5. **Activation Functions:** ReLU is used for non-linear transformations, and Softmax is applied to the output for classification.

---

## Training Process

The training process involves:
1. Forward pass: Feeding the input data to the model.
2. Loss computation: Calculating the prediction error using `CrossEntropyLoss`.
3. Backward pass: Updating model weights using the Adam optimizer.
4. Progress logging: Loss values are printed every 20 batches for monitoring.

---

## Testing Process

The testing process evaluates the model's performance on the test dataset:
- **Metrics:** Average loss and classification accuracy.
- Predictions are compared with ground truth labels to calculate accuracy.

---

## Device Compatibility

The code automatically detects whether a GPU (CUDA) is available. If not, it defaults to the CPU for computations.

---

## Key Features

- **Simple CNN Architecture:** Easy to understand and modify.
- **Dynamic Batching:** Efficient handling of data during training and testing.
- **Device-Aware Training:** Utilizes GPU for faster computation if available.

---

## How to Use

1. Clone the repository.
2. Install the required libraries: `torch`, `torchvision`, and `matplotlib`.
3. Run the script to train the model:
   - Modify hyperparameters like `batch_size`, `learning_rate`, and `epochs` as needed.
4. View the training loss and accuracy during training and testing.

---

## Results

The model achieves high accuracy on the MNIST test dataset after training for a few epochs, making it suitable for digit classification tasks.

---

## Notes

- Experiment with different hyperparameters and architectures to improve performance.
- Ensure GPU support for faster training.
- This implementation is ideal for beginners looking to learn CNNs and PyTorch.

--- 

