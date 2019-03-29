from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, optimizers, layers, losses, models, callbacks, utils, regularizers
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

import utils


FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'maxes_path', default = None, help = 'The name of the files that contains the max values to be used for min-max normalization.')
flags.DEFINE_string(name = 'normalized_data_path', default = None, help = 'The path to the normalized data.')
flags.DEFINE_string(name = 'settings_file', default = 'grid_settings.txt', help = 'The settings of the grid search.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data and seed the random number generator.')
flags.DEFINE_integer(name = 'batch_size', default = 100, help = 'The batch size to use for training.')
flags.DEFINE_integer(name = 'max_epochs', default = 500, help = 'The maximum amount of epochs.')
flags.DEFINE_integer(name = 'experiments', default = 40, help = 'The amount of networks to train.')

FLAGS(sys.argv)


def train(model, train_data, train_labels, validation_data, validation_labels, batch_size, max_epochs):
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
    mc = callbacks.ModelCheckpoint('best_model')
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


def main(argv):
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = \
        utils.load_data_without_game_crossover(
            FLAGS.data_path, 
            0.7, 
            0.15, 
            0.15, 
            FLAGS.seed, 
            FLAGS.maxes_path, 
            FLAGS.normalized_data_path
        )

    input_size = train_data.shape[1]
    num_classes = train_labels.shape[1]

    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)

    min_hidden_layers = 0
    max_hidden_layers = 5
    min_neurons = 10
    max_neurons = 1000
    dropout_list = [True, False]
    regularization_list = [True, False]

    settings_seed = 'Seed: ' + str(FLAGS.seed)
    settings_batch_size = 'Batch size: ' + str(FLAGS.batch_size)
    settings_max_epochs = 'Max epochs: ' + str(FLAGS.max_epochs)
    settings_min_hidden_layers = 'min hidden layers: ' + str(min_hidden_layers)
    settings_max_hidden_layers = 'max hidden layers: ' + str(max_hidden_layers)
    settings_min_neurons = 'min neurons in hidden layers: ' + str(min_neurons)
    settings_max_neurons = 'max neurons in hidden layers: ' + str(max_neurons)
    settings_regularization = 'regularization: ' + str(regularization_list)
    settings_dropout = 'Dropout: ' + str(dropout_list)
    settings = settings_seed + '\n' + \
        settings_batch_size + '\n' + \
        settings_max_epochs + '\n' + \
        settings_min_hidden_layers + '\n' + \
        settings_max_hidden_layers + '\n' + \
        settings_min_neurons + '\n' + \
        settings_max_neurons + '\n' + \
        settings_regularization + '\n' + \
        settings_dropout
    
    with open(FLAGS.settings_file, 'w+') as f:
        f.write(settings)

    print('_____________________________________________________________________________________')
    print("Grid searching:")
    print('-------------------------------------------------------------------------------------')
    print(settings_min_hidden_layers)
    print(settings_max_hidden_layers)
    print(settings_min_neurons)
    print(settings_max_neurons)
    print(settings_dropout)
    print(settings_regularization)
    print(settings_seed)
    print(settings_batch_size)
    print(settings_max_epochs)
    print('-------------------------------------------------------------------------------------')

    try:
        for _ in range(FLAGS.experiments):
            hidden_layers = random.randint(min_hidden_layers, max_hidden_layers)
            neurons = random.randint(min_neurons, max_neurons)
            dropout = random.choice(dropout_list)
            regularization = random.choice(regularization_list)

            if dropout:
                dropout_variable = random.uniform(0.0, 0.1)

            if regularization:
                regularization_variable = random.uniform(0.0, 0.1)

            model_dir = 'l' + str(hidden_layers) + ' _n' + str(neurons) + ' _d' + str(dropout) + ' _r' + str(regularization)
            os.mkdir(model_dir)

            network_settings = \
                '# hidden layers: ' + str(hidden_layers) + '\n' + \
                '# neurons in hidden layers: ' + str(neurons) + '\n' + \
                (('Dropout: ' + str(dropout_variable)) if dropout else ('Dropout: ' + str(dropout))) + '\n' + \
                (('Regularization: ' + str(regularization_variable)) if regularization else ('Regularization: ' + str(regularization))) + '\n' + \
                '\n'

            model_results_file = os.path.join(model_dir, 'results.txt')

            with open(model_results_file, 'w+') as f:
                f.write(network_settings)

            print('=========================================================================')
            print('Training model with')
            print('# hidden layers: ' + str(hidden_layers))
            print('# neurons: ' + str(neurons))
            print(('Dropout: ' + str(dropout_variable)) if dropout else ('Dropout: ' + str(dropout)))
            print((('Regularization: ' + str(regularization_variable)) if regularization else ('Regularization: ' + str(regularization))))
            print('=========================================================================')
            
            model = keras.models.Sequential()

            model.add(layers.Dense(neurons, activation=tf.nn.relu, input_shape=(input_size,)))

            for _ in range(hidden_layers):
                if dropout:
                    model.add(layers.Dropout(dropout_variable))
                
                if regularization:
                    model.add(layers.Dense(neurons, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(regularization_variable)))
                else:
                    model.add(layers.Dense(neurons, activation=tf.nn.relu))

            if dropout:
                model.add(layers.Dropout(dropout_variable))

            model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))

            model.summary()

            model.compile(loss=losses.categorical_crossentropy,
                        optimizer=optimizers.Adam(),
                        metrics=[utils.top_1_categorical_accuracy, utils.top_3_categorical_accuracy])

            best_model_path = os.path.join(model_dir, 'best_model.h5')

            es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
            mc = callbacks.ModelCheckpoint(best_model_path, monitor='val_loss', mode='min', save_best_only=True)
            history = model.fit(
                train_data, 
                train_labels, 
                validation_data=(validation_data, validation_labels),
                epochs=FLAGS.max_epochs,
                batch_size=FLAGS.batch_size,
                verbose = 0,
                callbacks=[es, mc]
            )

            scores = model.evaluate(
                x=test_data, 
                y=test_labels, 
                batch_size=FLAGS.batch_size, 
                verbose=0
            )

            print('Final model scores:')

            results = ''
            for index in range(len(model.metrics_names)):
                results += '%s: %.2f%%' % (model.metrics_names[index], scores[index]*100)
                results += '\n'
                print('%s: %.2f%%' % (model.metrics_names[index], scores[index]*100))

            with open(model_results_file, 'a') as f:
                f.write(results)

            name = 'model.h5'
            path = os.path.join(model_dir, name)
            model.save(path)

            name = 'top_1'
            path = os.path.join(model_dir, name)
            np.save(path , history.history['top_1_categorical_accuracy'])

            name = 'val_top_1'
            path = os.path.join(model_dir, name)
            np.save(path , history.history['val_top_1_categorical_accuracy'])

            name = 'top_3'
            path = os.path.join(model_dir, name)
            np.save(path , history.history['top_3_categorical_accuracy'])

            name = 'val_top_3'
            path = os.path.join(model_dir, name)
            np.save(path , history.history['val_top_3_categorical_accuracy'])
            
            name = 'loss'
            path = os.path.join(model_dir, name)
            np.save(path , history.history['loss'])

            name = 'val_loss'
            path = os.path.join(model_dir, name)
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
            path = os.path.join(model_dir, name)
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
            path = os.path.join(model_dir, name)
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
            path = os.path.join(model_dir, name)
            plt.savefig(path)
            plt.close()

            print('=========================================================================')
            print('Finished with model')
            print('=========================================================================')
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    app.run(main)