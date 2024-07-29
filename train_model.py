# train_model.py
import struct
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

# Define the paths to the dataset files
train_images_path = 'path/to/train-images.idx3-ubyte'
train_labels_path = 'path/to/train-labels.idx1-ubyte'
test_images_path = 'path/to/t10k-images.idx3-ubyte'
test_labels_path = 'path/to/t10k-labels.idx1-ubyte'

# Helper function to read the dataset files
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Load the datasets
train_images = read_idx(train_images_path)
train_labels = read_idx(train_labels_path)
test_images = read_idx(test_images_path)
test_labels = read_idx(test_labels_path)

# Display the first image in the training set for verification
plt.imshow(train_images[0], cmap='gray')
plt.title(f'Label: {train_labels[0]}')
plt.show()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Save the model
model.save('app/mnist_model.h5')
