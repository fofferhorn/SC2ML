from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os

from absl import app
from absl import flags
import sys

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'train_path', default = 'split\\train', help = 'The path to the training data.')
flags.DEFINE_string(name = 'validation_path', default = 'split\\validation', help = 'The path to the validation data.')
flags.DEFINE_string(name = 'test_path', default = 'split\\test', help = 'The path to the test data.')

FLAGS(sys.argv)


def train(model, train_data, train_labels):
    model.fit(train_data, train_labels, epochs=10)

    return model


def validate(model, validation_data, validation_labels):
    pass


def load_training_data(train_path):
    train_data = []
    train_labels = []
    for _file in os.listdir(train_path):
        if _file.endswith('.npy'):
            for data_point in np.load(os.path.join(train_path, _file)):
                train_data.append(data_point[:-1])
                train_labels.append(data_point[-1])
        
    return np.array(train_data), np.array(train_labels)


def load_validation_data(validation_path):
    validation_data = []
    validation_labels = []
    for _file in os.listdir(validation_path):
        if _file.endswith('.npy'):
            for data_point in np.load(os.path.join(validation_path, _file)):
                validation_data.append(data_point[:-1])
                validation_labels.append(data_point[-1])
        
    return np.array(validation_data), np.array(validation_labels)


def load_test_data(test_path):
    test_data = []
    test_labels = []
    for _file in os.listdir(test_path):
        if _file.endswith('.npy'):
            for data_point in np.load(os.path.join(test_path, _file)):
                test_data.append(data_point[:-1])
                test_labels.append(data_point[-1])
        
    return np.array(test_data), np.array(test_labels)


def main(argv):
    print('Loading data...')
    train_data, train_labels = load_training_data(FLAGS.train_path)
    validation_data, validation_labels = load_validation_data(FLAGS.validation_path)
    test_data, test_labels = load_test_data(FLAGS.test_path)
    print('Data loaded.')

    input_layer = keras.layers.Input(shape=(194, ), name='input')
    hidden_layer_1 = keras.layers.Dense(1024, activation=tf.nn.relu)(input_layer)
    hidden_layer_2 = keras.layers.Dense(1024, activation=tf.nn.relu)(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(1024, activation=tf.nn.relu)(hidden_layer_2)
    output_layer = keras.layers.Dense(54, activation=tf.nn.softmax)(hidden_layer_3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    print('=========================================================================')
    print('Training model...')
    print('=========================================================================')

    try:
        while True:
            model = train(model, train_data, train_labels)
            validate(model, validation_data, validation_labels)
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    app.run(main)