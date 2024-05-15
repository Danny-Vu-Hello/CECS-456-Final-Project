import os
import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt


# function to preprocess the images
def preprocess_image(image, label):
    # Rescale pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# Path to custom training folder (same directory as python file)
train_data_dir = os.path.join(os.getcwd(), 'natural_images')

# Load the data from the directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(28, 28),
    batch_size=256)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(28, 28),
    batch_size=256)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    seed=42,
    image_size=(28, 28),
    batch_size=256)

# Apply rescaling to the images using the preprocessing_function parameter
train_ds = train_ds.map(preprocess_image)
val_ds = val_ds.map(preprocess_image)
test_ds = test_ds.map(preprocess_image)

# prefetch the datasets for better performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

X_train_full, y_train_full = [], []
for images, labels in train_ds:
    X_train_full.append(images.numpy())
    y_train_full.append(labels.numpy())
X_train_full = np.concatenate(X_train_full)
y_train_full = np.concatenate(y_train_full)

X_valid, y_valid = [], []
for images, labels in val_ds:
    X_valid.append(images.numpy())
    y_valid.append(labels.numpy())
X_valid = np.concatenate(X_valid)
y_valid = np.concatenate(y_valid)

X_test, y_test = [], []
for images, labels in test_ds:
    X_test.append(images.numpy())
    y_test.append(labels.numpy())
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

X_train_full.shape

y_train_full

# Build CNN model
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu", padding="same", input_shape=[28, 28, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=64, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=8, activation="softmax"))  # 8 classes

cnn.summary()

# Compile the model
cnn.compile(loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = cnn.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[early_stopping])

# Evaluate the model on the test data
score = cnn.evaluate(test_ds)

print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])

X_new = X_test[:3]
y_proba = cnn.predict(X_new)
y_proba.round(2)

# y_pred = model.predict_classes(X_new) # deprecated
y_pred = np.argmax(cnn.predict(X_new), axis=-1)
y_pred

class_names = ["airplane", "car", "cat", "dog", "flower", "fruit", "motorbike", "person"]

np.array(class_names)[y_pred]

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
# save_fig('fashion_mnist_images_plot', tight_layout=False)
plt.show()
