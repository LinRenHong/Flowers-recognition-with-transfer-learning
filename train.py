# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub


# This function will split dataset directory to train directory and val directory.
def splitDateSet( base_dir, validation_split, classes ):
    for cl in classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        num_train = int(round(len( images) * (1.0 - validation_split)))
        train, val = images[:num_train], images[num_train:]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            shutil.move(v, os.path.join(base_dir, 'val', cl))


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# This function will plot training and validation accuracy/loss graphs
def plotTrainingAndValidationGraphs( history, epochs ):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# DataSet directory
dataSet_dir = r'/flower_photos'
base_dir = os.getcwd() + dataSet_dir
print("Path of dataset : {0}".format(base_dir))

# Five flowers
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
num_classes = len(classes)

# Let my dataSet 80% to train, 20% to validation
splitDateSet(base_dir=base_dir, validation_split=0.2, classes=classes)

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')


# Set parameter
batch_size = 32
IMG_SHAPE = 224

# Training set and image Augmentation
image_gen_train = ImageDataGenerator(
                                        rescale=1./255,
                                        rotation_range=45,
                                        width_shift_range=.15,
                                        height_shift_range=.15,
                                        horizontal_flip=True,
                                        zoom_range=0.5
                                    )


train_data_gen = image_gen_train.flow_from_directory(
                                                        batch_size=batch_size,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(IMG_SHAPE,IMG_SHAPE),
                                                        class_mode='sparse'
                                                    )


# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)


# Validation set
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(
                                                  batch_size=batch_size,
                                                  directory=val_dir,
                                                  target_size=(IMG_SHAPE, IMG_SHAPE),
                                                  class_mode='sparse'
                                                )

print("class labels is {0}".format(train_data_gen.class_indices))
# See Image batch and label batch
sample_training_images, label = next(train_data_gen)
print("Shape of image batch : {0}".format(sample_training_images.shape))
print("Label batch : {0}".format(label))

# Transfer Learning
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMG_SHAPE, IMG_SHAPE,3))

# Freeze the pre-train model
feature_extractor.trainable = False


model = tf.keras.Sequential([
                              feature_extractor,
                              layers.Dense(num_classes, activation='softmax')
                            ])

model.summary()

model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
             )

# # Path of saving checkpoint
# checkpoint_path = "training_model_checkpoint/flowers.ckpt"

# save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

EPOCHS = 15

# Training
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))),
    # callbacks=[save_checkpoint]
)



model_weights_name = r'another_flowers_recognition_model_weights.h5'

# Save the model weights
print("Saving model weights: {0}".format(model_weights_name))
model.save_weights(model_weights_name)
print("Success!")

# Plot Training and validation accuracy/loss graphs
plotTrainingAndValidationGraphs(history=history, epochs=EPOCHS)
