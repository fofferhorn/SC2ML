from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, optimizers, layers, losses, models, callbacks
from tensorflow.keras import backend as K


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

import os

from absl import app
from absl import flags
import sys


FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'model_name', default = 'model', help = 'The name to save the model with.')
flags.DEFINE_string(name = 'settings_file', default = 'grid_settings.txt', help = 'The settings of the grid search.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data and seed the random number generator.')
flags.DEFINE_integer(name = 'batch_size', default = 100, help = 'The batch size to use for training.')
flags.DEFINE_integer(name = 'max_epochs', default = 500, help = 'The maximum amount of epochs.')

FLAGS(sys.argv)


def train(model, train_data, train_labels, validation_data, validation_labels, batch_size, max_epochs):
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
    history = model.fit(
        train_data, 
        train_labels, 
        validation_data=(validation_data, validation_labels),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose = 0,
        callbacks=[es]
    )

    return model, history


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


def top_3_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_1_categorical_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def main(argv):
    cwd = os.getcwd()
    
    print('Loading data...')
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_data(FLAGS.data_path, 0.7, 0.2, 0.1, FLAGS.seed)
    print('Data loaded.')

    input_size = train_data.shape[1]
    num_classes = train_labels.shape[1]

    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)

    model_hidden_layers_list = [2, 3, 4]
    hidden_layer_neurons_list = [random.randint(50, 1000) for _ in range(5)]
    hidden_layer_neurons_list.sort()
    dropout_list = [True, False]


    settings_seed = 'Seed: ' + str(FLAGS.seed)
    settings_batch_size = 'Batch size: ' + str(FLAGS.batch_size)
    settings_max_epochs = 'Max epochs: ' + str(FLAGS.max_epochs)
    settings_hidden_layers = '# hidden layers: ' + str(model_hidden_layers_list)
    settings_neurons = '# neurons in hidden layers: ' + str(hidden_layer_neurons_list)
    settings_dropout = 'Dropout: ' + str(dropout_list)
    settings = settings_seed + '\n' + settings_batch_size + '\n' + settings_max_epochs + '\n' + settings_hidden_layers + '\n' + settings_neurons + '\n' + settings_dropout
    
    with open(FLAGS.settings_file, 'w+') as f:
        f.write(settings)

    print('_____________________________________________________________________________________')
    print("Grid searching:")
    print('-------------------------------------------------------------------------------------')
    print(settings_hidden_layers)
    print(settings_neurons)
    print(settings_dropout)
    print(settings_seed)
    print(settings_batch_size)
    print(settings_max_epochs)
    print('-------------------------------------------------------------------------------------')

    try:
        for model_hidden_layers in model_hidden_layers_list:
            for hidden_layer_neurons in hidden_layer_neurons_list:
                for dropout in dropout_list:
                    
                    # Build network
                    model = keras.models.Sequential()

                    model.add(layers.Dense(hidden_layer_neurons, activation=tf.nn.relu, input_shape=(input_size,)))
                    if dropout:
                        model.add(layers.Dropout(0.2))

                    for _ in range(model_hidden_layers-1):
                        model.add(layers.Dense(hidden_layer_neurons, activation=tf.nn.relu))
                        if dropout:
                            model.add(layers.Dropout(0.2))
                        
                    model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))

                    model.summary()

                    model.compile(loss=losses.categorical_crossentropy,
                                optimizer=optimizers.Adam(),
                                metrics=[top_1_categorical_accuracy, top_3_categorical_accuracy])

                    print('=========================================================================')
                    print('Training model with l=' + str(model_hidden_layers) + ' n=' + str(hidden_layer_neurons) + ' d=' + str(dropout) + '...')
                    print('=========================================================================')

                    model, history = train(
                        model, 
                        train_data, 
                        train_labels, 
                        validation_data, 
                        validation_labels,
                        FLAGS.batch_size,
                        FLAGS.max_epochs
                    )

                    model_dir = 'l' + str(model_hidden_layers) + '_n' + str(hidden_layer_neurons) + '_d' + str(dropout)

                    scores = model.evaluate(
                        x=test_data, 
                        y=test_labels, 
                        batch_size=FLAGS.batch_size, 
                        verbose=0
                    )

                    print('Final model scores:')

                    for index in range(len(model.metrics_names)):
                        print('%s: %.2f%%' % (model.metrics_names[index], scores[index]*100))

                    os.mkdir(model_dir)

                    name = FLAGS.model_name + 'h5'
                    path = os.path.join(cwd, model_dir, name)
                    model.save(path)

                    name = 'top_1'
                    path = os.path.join(cwd, model_dir, name)
                    np.save(path , history.history['top_1_categorical_accuracy'])

                    name = 'val_top_1'
                    path = os.path.join(cwd, model_dir, name)
                    np.save(path , history.history['val_top_1_categorical_accuracy'])

                    name = 'top_3'
                    path = os.path.join(cwd, model_dir, name)
                    np.save(path , history.history['top_3_categorical_accuracy'])

                    name = 'val_top_3'
                    path = os.path.join(cwd, model_dir, name)
                    np.save(path , history.history['val_top_3_categorical_accuracy'])
                    
                    name = 'loss'
                    path = os.path.join(cwd, model_dir, name)
                    np.save(path , history.history['loss'])

                    name = 'val_loss'
                    path = os.path.join(cwd, model_dir, name)
                    np.save(path , history.history['val_loss'])

                    # Plot training & validation top-1 accuracy values
                    plt.figure()
                    plt.plot(history.history['top_1_categorical_accuracy'])
                    plt.plot(history.history['val_top_1_categorical_accuracy'])
                    plt.title('Top-1 Model Accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc='upper left')
                    name = 'top_1.png'
                    path = os.path.join(cwd, model_dir, name)
                    plt.savefig(path)
                    plt.close()

                    # Plot training & validation top-3 accuracy values
                    plt.figure()
                    plt.plot(history.history['top_3_categorical_accuracy'])
                    plt.plot(history.history['val_top_3_categorical_accuracy'])
                    plt.title('Top-3 Model Accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc='upper left')
                    name = 'top_3.png'
                    path = os.path.join(cwd, model_dir, name)
                    plt.savefig(path)
                    plt.close()

                    # Plot training & validation loss values
                    plt.figure()
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('Model loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Test'], loc='upper left')
                    name = 'loss.png'
                    path = os.path.join(cwd, model_dir, name)
                    plt.savefig(path)
                    plt.close()

                    print('=========================================================================')
                    print('Finished with model with l=' + str(model_hidden_layers) + ' n=' + str(hidden_layer_neurons) + ' d=' + str(dropout))
                    print('=========================================================================')
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    app.run(main)