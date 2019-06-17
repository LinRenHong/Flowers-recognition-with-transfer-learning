# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras import layers

import tensorflow_hub as hub

from PIL import Image


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


def plotModelPrediction(image_path):
    image = Image.open(image_path)
    class_names = np.array([i for i in val_data_gen.class_indices])
    image = image.resize((IMG_SHAPE, IMG_SHAPE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255
    img = np.expand_dims(image, axis=0)
    predicted_result = model.predict(img)[0]
    predicted_id = np.argmax(predicted_result, axis=-1)

    confidence = predicted_result[predicted_id]
    print("Confidence: {0}".format(confidence))

    predicted_class_name = class_names[predicted_id]
    print(predicted_class_name)
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    plt.title(predicted_class_name.title(), color='blue')
    confidence_text = "Confidence: {0:%}".format(confidence)
    plt.text(112, 240, confidence_text, fontsize=18, ha='center', color='green')
    plt.axis('off')
    plt.suptitle("Model prediction")
    plt.show()


# This function will plot predictions of model
def plotModelPredictions(data_gen, model):
    class_names = np.array([i for i in data_gen.class_indices])

    print("Class names: {0}".format(class_names))

    image_batch, label_batch = next(data_gen)

    predicted_batch = model.predict(image_batch)

    predicted_ids = np.argmax(predicted_batch, axis=-1)
    predicted_class_names = class_names[predicted_ids]
    print("Predicted class names: {0}".format(predicted_class_names))

    print("Labels: ", label_batch)
    print("Predicted labels: ", predicted_ids)

    plt.figure(figsize=(10, 9))
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        color = "blue" if predicted_ids[n] == label_batch[n] else "red"
        plt.title(predicted_class_names[n].title(), color=color)
        plt.axis('off')
        plt.suptitle("Model predictions (blue: correct, red: incorrect)")
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

# Validation set
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(
                                                  batch_size=batch_size,
                                                  directory=val_dir,
                                                  target_size=(IMG_SHAPE, IMG_SHAPE),
                                                  class_mode='sparse'
                                                )

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# Transfer Learning
# URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
# feature_extractor = hub.KerasLayer(URL,
#                                    input_shape=(IMG_SHAPE, IMG_SHAPE,3))


PRE_TRAIN_MODEL_PATH = r"/Users/linrenhong/Documents/Mac Program/Python/CNN/Flowers_Recognition_With_CNN/pre_train_model"
feature_extractor = hub.KerasLayer(PRE_TRAIN_MODEL_PATH,
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

model_weights_name = r'flowers_recognition_model_weights.h5'

print("Loading model weights: {0}".format(model_weights_name))
model.load_weights(model_weights_name)
print("Success!")


# Use model to predict Validation set ( random 30 images )
plotModelPredictions(data_gen=val_data_gen, model=model)

# Use model to predict your own image
# image_path = r'YOUR_IMAGE_PATH''
# plotModelPrediction(image_path)






