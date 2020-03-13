import os
from PIL import Image
import shutil
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf  
import random
import pickle


_TRAINING_LOCATION = '../dataset/cat-and-dog/processed_merged_training_set/'
_TESTING_LOCATION = '../dataset/cat-and-dog/processed_merged_test_set/'
_MODEL_FILE_NAME = '../trained_model/keras__cat_dog_model.h5'
_LABEL = {
    'dog' : 0,
    'cat' : 1
}

def process_images(image_source, 
                    processed_image_destination,
                    base_width):
    """
        taking images from image source and 
        process them and save in a destination
        directory
    """
    shutil.rmtree(processed_image_destination, ignore_errors=True)
    os.mkdir(processed_image_destination)

    for file in os.listdir(image_source):
        print(f'processing image file {file}')
        img = Image.open(image_source + file)
        img = img.resize((base_width, base_width), Image.ANTIALIAS)
        img.save(processed_image_destination + file) 


def image_to_array(image_file):
    image_as_array = np.asarray(Image.open(image_file)) 

    return image_as_array


def array_to_image(image_array):
    plt.imsave('dummp.jpeg', image_array , cmap='Greys')


def get_image_arrays_from_files(directory):
    files = []
    labels = []
    for file in os.listdir(directory):
        label = file[:file.find('.')]
        files.append(file)
        labels.append(_LABEL[label])
    
    image_array = []
    for file in files:
        image_arr = image_to_array(directory + file)
        image_array.append(image_arr)

    image_array = np.array(image_array).astype(float) / 255.0
    labels = np.array(labels).astype(float)

    return image_array, labels 
    

def dumb_neural_network_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(200, 200, 3)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    model.compile(optimizer='adam',
        loss= 'sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def keras_image_net_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model


if __name__ == "__main__":

    training_data, training_labels = get_image_arrays_from_files(_TRAINING_LOCATION)
    test_data, test_labels  = get_image_arrays_from_files(_TESTING_LOCATION)

    model = keras_image_net_model()
    model.fit(training_data, training_labels, epochs=10)
    model.save(_MODEL_FILE_NAME)

    model = tf.keras.models.load_model(_MODEL_FILE_NAME)

    predict = model.predict_classes(test_data)
    accuracy = sum([1 if predict[i] == test_labels[i] else 0 for i in range(len(test_labels))]) / len(test_labels)
    print(f'accuracy = {accuracy * 100}')



































 # process_images('../cat-and-dog/training_set/merged_training_set/', 
    #                 '../cat-and-dog/training_set/processed_merged_training_set/', 
    #                 base_width=200)

    # process_images('../cat-and-dog/test_set/merged_test_set/', 
    #                 '../cat-and-dog/test_set/processed_merged_test_set/', 
    #                 base_width=200)

    # image_arr = image_to_array('../processed_images/4.jpg') / 255.0



