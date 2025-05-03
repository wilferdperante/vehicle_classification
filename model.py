import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Filter only classes: 0 (airplane), 1 (automobile), 8 (ship), 9 (truck)
desired_classes = [0, 1, 8, 9]
class_names = ['plane', 'car', 'ship', 'truck']

# Create filter mask
train_filter = np.isin(train_labels, desired_classes).flatten()
test_filter = np.isin(test_labels, desired_classes).flatten()

# Apply filter
train_images = train_images[train_filter]
train_labels = train_labels[train_filter]
test_images = test_images[test_filter]
test_labels = test_labels[test_filter]

# Remap labels to 0–3
label_map = {0: 0, 1: 1, 8: 2, 9: 3}
train_labels = np.vectorize(label_map.get)(train_labels)
test_labels = np.vectorize(label_map.get)(test_labels)

# Show sample images
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
    plt.xlabel(class_names[int(train_labels[i])])
plt.show()

# Limit dataset for faster training
train_images = train_images[:20000]
train_labels = train_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

# Build CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  # 4 classes

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the trained model
model.save('image_classifier.keras')
