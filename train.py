import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions

# from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

from tensorflow.keras.utils import image_dataset_from_directory

import pathlib



data_dir = 'data'
data_dir = pathlib.Path(data_dir)



input_size = 299
batch_size = 32
learning_rate = 0.001
size = 500
droprate = 0.8

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=50,
    image_size=(input_size, input_size),
    batch_size=batch_size)

val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=50,
  image_size=(input_size, input_size),
  batch_size=batch_size)


# defineing a model function
def make_model(learning_rate=0.01, size_inner=100, droprate=0.5, input_size=150):
    resize_and_rescale = keras.Sequential([
    keras.layers.Resizing(input_size, input_size,),
    keras.layers.Rescaling(1./255)])

    data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal",
                      input_shape=(input_size,
                                  input_size,
                                  3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),])


    model = keras.models.Sequential()
    model.add(resize_and_rescale)
    model.add(data_augmentation)
    #model.add(keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
    model.add(keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Dropout(droprate))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(size_inner, activation='relu'))
    model.add(keras.layers.Dense(3))
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


checkpoint = keras.callbacks.ModelCheckpoint(
    'xray_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

# training the model
history = model.fit(train_ds, epochs=20, validation_data=val_ds,
                   callbacks=[checkpoint])


model_name = 'xray_v1_15_0.913.h5' # xray_v1_15_0.913.h5 is the model with best accuracy
model_best = keras.models.load_model(model_name)

# Convert the model to tflite.
converter = tf.lite.TFLiteConverter.from_keras_model(model_best)
tflite_model = converter.convert()

# Save the model.
with open('xray_model.tflite', 'wb') as f:
  f.write(tflite_model)