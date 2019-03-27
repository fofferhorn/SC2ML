from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import metrics, optimizers, layers, losses, models, callbacks, utils
from tensorflow.keras import backend as K


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math

import os

from absl import app
from absl import flags
import sys

import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'model_name', default = 'model', help = 'The name to save the model with.')
flags.DEFINE_string(name = 'maxes_path', default = None, help = 'The name of the files that contains the max values to be used for min-max normalization.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data.')
flags.DEFINE_integer(name = 'batch_size', default = 100, help = 'The batch size to use for training.')
flags.DEFINE_integer(name = 'max_epochs', default = 500, help = 'The maximum amount of epochs.')

FLAGS(sys.argv)


def train(model, train_data, train_labels, validation_data, validation_labels, batch_size, max_epochs):
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
    history = model.fit(
        train_data, 
        train_labels, 
        validation_data=(validation_data, validation_labels),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose = 1,
        callbacks=[es]
    )

    return model, history


def load_data(data_path, train, validation, test, seed = None, maxes_path = None):
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
    for data_point in data_and_labels[:len(data_and_labels)]:
        if len(data_point) == 248:
            data.append(data_point[:-54])
            labels.append(data_point[-54:])

    print('Normalizing data...')
    data = min_max_norm(data, maxes_path)
    print('Data normalized.')

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

    print('_____________________________________________________________________________________')
    print('Data meta data')
    print('{:20s} {:7d}'.format('# of games', len(data_paths)))
    print('{:20s} {:7d}'.format('# of data points', len(data_and_labels)))
    print('Split seed: ' + str(seed))
    print('-------------------------------------------------------------------------------------')
    print('| {:25s} | {:25s} | {:25s} |'.format('Data', '# data points', '# data point dimensions'))
    print('|---------------------------|---------------------------|---------------------------|')
    print('| {:25s} | {:25d} | {:25d} |'.format('train_data shape', len(train_data), len(train_data[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('train_labels shape', len(train_labels), len(train_labels[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('validation_data shape', len(validation_data), len(validation_data[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('validation_labels shape', len(validation_labels), len(validation_labels[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('test_data shape', len(test_data), len(test_data[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('test_labels shape', len(test_labels), len(test_labels[0])))
    print('-------------------------------------------------------------------------------------')

    return np.array(train_data), np.array(train_labels), np.array(validation_data), np.array(validation_labels), np.array(test_data), np.array(test_labels)


def load_data_2(data_path, train, validation, test, seed = None, maxes_path = None):
    # Make list of the paths to all the replays
    cwd = os.getcwd()
    data_paths = []
    data_base_path = os.path.join(cwd, data_path)
    for data in os.listdir(data_path):
        _data_path = os.path.join(data_base_path, data)
        if os.path.isfile(_data_path) and data.lower().endswith('.npy'):
            data_paths.append(_data_path)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(data_paths)

    train_end = int(len(data_paths) * train)
    validation_end = int(len(data_paths) * (train + validation))

    train_paths = []
    for index in range(train_end):
        train_paths.append(data_paths[index])
    
    validation_paths = []
    for index in range(train_end, validation_end):
        validation_paths.append(data_paths[index])

    test_paths = []
    for index in range(validation_end, len(data_paths)):
        test_paths.append(data_paths[index])

    train_end = int(len(data) * train)
    validation_end = int(len(data) * (train + validation))

    train_data = []
    for path in train_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                train_data.append(0)

    validation_data = []
    for path in validation_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                validation_data.append(0)

    test_data = []
    for path in test_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                test_data.append(0)
    
    data = []
    labels = []
    for path in data_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                data.append(data_point[:-54])
                labels.append(data_point[-54:])

    print('Normalizing data...')
    data = min_max_norm(data, maxes_path)
    print('Data normalized.')
    
    train_end = len(test_data)
    validation_end = len(test_data) + len(validation_data)

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

    print('_____________________________________________________________________________________')
    print('Data meta data')
    print('{:20s} {:7d}'.format('# of games', len(data_paths)))
    print('{:20s} {:7d}'.format('# of data points', len(data)))
    print('Split seed: ' + str(seed))
    print('-------------------------------------------------------------------------------------')
    print('| {:25s} | {:25s} | {:25s} |'.format('Data', '# data points', '# data point dimensions'))
    print('|---------------------------|---------------------------|---------------------------|')
    print('| {:25s} | {:25d} | {:25d} |'.format('train_data shape', len(train_data), len(train_data[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('train_labels shape', len(train_labels), len(train_labels[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('validation_data shape', len(validation_data), len(validation_data[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('validation_labels shape', len(validation_labels), len(validation_labels[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('test_data shape', len(test_data), len(test_data[0])))
    print('| {:25s} | {:25d} | {:25d} |'.format('test_labels shape', len(test_labels), len(test_labels[0])))
    print('-------------------------------------------------------------------------------------')

    return np.array(train_data), np.array(train_labels), np.array(validation_data), np.array(validation_labels), np.array(test_data), np.array(test_labels)


def min_max_norm(data, maxes_path = None):
    if maxes_path is None:
        maxes = []
        for i in range(len(data[0])):
            max = 0
            for j in range(len(data)):
                if data[j][i] > max:
                    max = data[j][i]
            maxes.append(max)

        np.array(maxes)
        np.savetxt('maxes.txt', maxes)
    else:
        maxes = np.loadtxt(maxes_path)

    norm_data = []
    for i in range(len(data)):
        norm_data_point = []
        for j in range(len(data[i])):
            if maxes[j] == 0:
                norm_data_point.append(0)
            else:
                norm_value = data[i][j]/maxes[j]
                if math.isnan(norm_value):
                    norm_data_point.append(0)
                else:
                    norm_data_point.append(norm_value)
        norm_data.append(norm_data_point)

    return norm_data
    

def top_3_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_1_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def main(argv):
    print('Loading data...')
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_data_2(FLAGS.data_path, 0.7, 0.2, 0.1, FLAGS.seed, FLAGS.maxes_path)
    print('Data loaded.')

    input_size = train_data.shape[1]
    num_classes = train_labels.shape[1]

    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)

    model = models.Sequential()
    model.add(layers.Dense(331, activation=tf.nn.relu, input_shape=(input_size,)))
    model.add(layers.Dense(331, activation=tf.nn.relu))
    model.add(layers.Dense(num_classes, activation=tf.nn.softmax))

    model.summary()

    model.compile(loss=losses.categorical_crossentropy,
                optimizer=optimizers.Adam(),
                metrics=[top_1_categorical_accuracy, top_3_categorical_accuracy])
    
    print('=========================================================================')
    print('Training model...')
    print('=========================================================================')

    try:
        model, history = train(
            model, 
            train_data, 
            train_labels, 
            validation_data, 
            validation_labels,
            FLAGS.batch_size,
            FLAGS.max_epochs
        )

        model.save(FLAGS.model_name + '.h5')

        scores = model.evaluate(
            x=test_data, 
            y=test_labels, 
            batch_size=FLAGS.batch_size, 
            verbose=1
        )

        print('=========================================================================')
        print('Finished training model.')
        print('=========================================================================')

        print('Calculating model scores...')
        print('Model scores:')

        for index in range(len(model.metrics_names)):
            print('%s: %.2f%%' % (model.metrics_names[index], scores[index]*100))

        print()

        print('Making plots...')

        # Plot training & validation top-1 accuracy values
        plt.plot(history.history['top_1_categorical_accuracy'])
        plt.plot(history.history['val_top_1_categorical_accuracy'])
        plt.title('Top-1 Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.close()

        # Plot training & validation top-3 accuracy values
        plt.plot(history.history['top_3_categorical_accuracy'])
        plt.plot(history.history['val_top_3_categorical_accuracy'])
        plt.title('Top-3 Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.close()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.close()
        
    except KeyboardInterrupt:
        model.save(FLAGS.model_name + '_interrupted.h5')
        return
    except KeyboardInterrupt:
        model.save(FLAGS.model_name + '_error.h5')
        return


if __name__ == "__main__":
    app.run(main)