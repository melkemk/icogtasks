import numpy as np
from PIL import Image
import os
import gzip
import torch

# Load the MNIST dataset images
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

# Load the MNIST dataset labels
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

data_path_images = '/home/melek/icog/gvae/data/MNIST/raw/train-images-idx3-ubyte.gz'
data_path_labels = '/home/melek/icog/gvae/data/MNIST/raw/train-labels-idx1-ubyte.gz'

images = load_mnist_images(data_path_images)
labels = load_mnist_labels(data_path_labels)

# Read the first image and label
first_image = images
first_label = labels
 
# Convert the image to a PyTorch tensor
tensor_image = torch.tensor(first_image, dtype=torch.float32)

# Display the tensor's parameters
print(f"Tensor shape: {tensor_image.shape}")
# print(f"Tensor dtype: {tensor_image.dtype}")
print(tensor_image)

# Print the first label
print(f"First label: {first_label.shape}")

