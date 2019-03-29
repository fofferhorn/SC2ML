from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import metrics, optimizers, layers, losses, models, callbacks, utils, regularizers
from tensorflow.keras import backend as K


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

from absl import app
from absl import flags
import sys
import os

import numpy as np

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'model_name', default = 'model', help = 'The name to save the model with.')
flags.DEFINE_string(name = 'maxes_path', default = None, help = 'The name of the files that contains the max values to be used for min-max normalization.')
flags.DEFINE_string(name = 'normalized_data_path', default = None, help = 'The path to the normalized data.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data.')
flags.DEFINE_integer(name = 'batch_size', default = 100, help = 'The batch size to use for training.')
flags.DEFINE_integer(name = 'max_epochs', default = 100, help = 'The maximum amount of epochs.')

FLAGS(sys.argv)


def train(model, train_data, train_labels, validation_data, validation_labels, batch_size, max_epochs):
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
    history = model.fit(
        train_data, 
        train_labels, 
        validation_data=(validation_data, validation_labels),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose = 1
        # callbacks=[es]
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

    for i in range(3):
        print(train_data[i])

    input_size = train_data.shape[1]
    num_classes = train_labels.shape[1]

    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)

    model = models.Sequential()
    model.add(layers.Dense(500, activation=tf.nn.relu, input_shape=(input_size,)))
    model.add(layers.Dense(500, activation=tf.nn.relu))
    model.add(layers.Dense(500, activation=tf.nn.relu))
    model.add(layers.Dense(num_classes, activation=tf.nn.softmax))

    model.summary()

    model.compile(loss=losses.categorical_crossentropy,
                optimizer=optimizers.Adam(),
                metrics=[utils.top_1_categorical_accuracy, utils.top_3_categorical_accuracy])
    
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