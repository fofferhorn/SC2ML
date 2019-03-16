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

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'model_name', default = 'model', help = 'The name to save the model with.')
flags.DEFINE_boolean(name = 'resume_from_model', default = False, help = 'Continue training a model.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data.')
flags.DEFINE_integer(name = 'batch_size', default = 32, help = 'The batch size to use for training.')

FLAGS(sys.argv)


def train(model, train_data, train_labels, validation_data, validation_labels, batch_size):
    model.fit(
        train_data, 
        train_labels, 
        validation_data=(validation_data, validation_labels),
        epochs=1000,
        batch_size=batch_size
    )

    return model


def validate(model, validation_data, validation_labels):
    pass


def load_data(data_path, train, validation, test, seed = None):
    # Make list of the paths to all the replays
    cwd = os.getcwd()
    data_paths = []
    data_base_path = os.path.join(cwd, data_path)
    for data in os.listdir(data_path):
        _data_path = os.path.join(data_base_path, data)
        if os.path.isfile(_data_path) and data.lower().endswith('.npy'):
            data_paths.append(_data_path)

    data_and_labels = []
    for path in data_paths:
        for data_point in np.load(path):
            data_and_labels.append(data_point)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(data_and_labels)

    data = []
    labels = []
    for data_point in data_and_labels:
        data.append(data_point[:-54])
        labels.append(data_point[-54:])

    data = keras.utils.normalize(data, axis=-1, order=2)

    train_end = int(len(data) * train)
    validation_end = int(len(data) * (train + validation))

    train_data = []
    train_labels = []
    for index in range(train_end):
            train_data.append(data[index])
            train_labels.append(labels[index])

    validation_data = []
    validation_labels = []
    for index in range(train_end, validation_end):
            validation_data.append(data[index])
            validation_labels.append(labels[index])

    test_data = []
    test_labels = []
    for index in range(validation_end, len(data)):
            test_data.append(data[index])
            test_labels.append(labels[index])

    return np.array(train_data), np.array(train_labels), np.array(validation_data), np.array(validation_labels), np.array(test_data), np.array(test_labels)


def main(argv):
    print('Loading data...')
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_data(FLAGS.data_path, 0.7, 0.2, 0.1, FLAGS.seed)
    print('Data loaded.')

    print('train_data: ' + str(train_data.shape))
    print('train_labels: ' + str(train_labels.shape))
    print('validation_data: ' + str(validation_data.shape))
    print('validation_labels: ' + str(validation_labels.shape))
    print('test_data: ' + str(test_data.shape))
    print('test_labels: ' + str(test_labels.shape))
    
    if not FLAGS.resume_from_model:
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(train_data.shape[1],)))
        model.add(keras.layers.Dense(train_labels.shape[1], activation=tf.nn.softmax))

        print('model input: ' + str(model.layers[0].input.shape))
        print('model output: ' + str(model.layers[-1].output.shape))

        model.summary()

        model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        
        print('=========================================================================')
        print('Training model...')
        print('=========================================================================')

        batch = 0

        try:
            while True:
                model = train(
                    model, 
                    train_data, 
                    train_labels, 
                    validation_data, 
                    validation_labels,
                    FLAGS.batch_size
                )

                # Backup the model.
                model.save(FLAGS.model_name + '_' + batch + '.h5')
                batch += 1
        except KeyboardInterrupt:
            model.save(FLAGS.model_name + '_interrupted.h5')
            return
        # except KeyboardInterrupt:
        #     model.save(FLAGS.model_name + '_error.h5')
        #     return

    else:
        model = keras.models.load_model(FLAGS.model_name)



if __name__ == "__main__":
    app.run(main)







# def load_training_data(train_path):
#     train_data = []
#     train_labels = []
#     for _file in os.listdir(train_path):
#         if _file.endswith('.npy'):
#             for data_point in np.load(os.path.join(train_path, _file)):
#                 train_data.append(data_point[:-1])
#                 train_labels.append(data_point[-1])
        
#     return np.array(train_data), np.array(train_labels)


# def load_validation_data(validation_path):
#     validation_data = []
#     validation_labels = []
#     for _file in os.listdir(validation_path):
#         if _file.endswith('.npy'):
#             for data_point in np.load(os.path.join(validation_path, _file)):
#                 validation_data.append(data_point[:-1])
#                 validation_labels.append(data_point[-1])
        
#     return np.array(validation_data), np.array(validation_labels)


# def load_test_data(test_path):
#     test_data = []
#     test_labels = []
#     for _file in os.listdir(test_path):
#         if _file.endswith('.npy'):
#             for data_point in np.load(os.path.join(test_path, _file)):
#                 test_data.append(data_point[:-1])
#                 test_labels.append(data_point[-1])
        
#     return np.array(test_data), np.array(test_labels)