# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, optimizers, layers, losses, models
from tensorflow.keras import backend as K

import os

from absl import app
from absl import flags
import sys

import numpy as np

import constants as c


FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'model_name', default = 'model', help = 'The name to save the model with.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data.')

FLAGS(sys.argv)


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


if __name__ == "__main__":
    print('Loading data...')
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_data(FLAGS.data_path, 0.7, 0.2, 0.1, FLAGS.seed)
    print('Data loaded.')    

    model = models.load_model(FLAGS.model_name, {"top_1_categorical_accuracy": top_1_categorical_accuracy, "top_3_categorical_accuracy": top_3_categorical_accuracy})

    test_size = 300000

    test_data = train_data[:test_size]
    test_labels = train_labels[:test_size]

    test_predictions = model.predict(test_data, verbose = 0)
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    correct_classifications = 0

    for i in range(len(test_predictions)):
        prediction = np.amax(test_predictions[i])
        prediction_action = c.protoss_macro_actions[np.argmax(test_predictions[i])]

        actual = np.amax(test_labels[i])
        actual_action = c.protoss_macro_actions[np.argmax(test_labels[i])]
        
        if prediction_action == actual_action:
            correct_classifications += 1

        # print('_____________________________________________________________________________________')
        # print('{:17s} - {:3f} - {:20s}'.format('Max prediction ', prediction, prediction_action))
        # print('{:17s} - {:3f} - {:20s}'.format('Max actual ', actual, actual_action))
        # print('_____________________________________________________________________________________')

    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('| {:30s} | {:12s} | {:18s} | {:12s} | {:22s} | {:22s} | {:22s} |'.format('Action', '# actual', 'Actual % of total', '# predicted', 'Predicted % of total', '# correctly predicted', '% correctly predicted' ))
    print('|--------------------------------|--------------|--------------------|--------------|------------------------|------------------------|------------------------|')
    for i in range(len(c.protoss_macro_actions)):
        action_name = c.protoss_macro_actions[i]

        actual = 0
        for j in range(len(test_labels)):
            if np.argmax(test_labels[j]) == i:
                actual += 1
        
        percentage_actual = actual/test_size*100

        predicted = 0
        correctly_predicted = 0
        for j in range(len(test_predictions)):
            if np.argmax(test_predictions[j]) == i:
                predicted += 1
            
                if np.argmax(test_labels[j]) == i:
                    correctly_predicted += 1

        percentage_predicted = predicted/test_size*100
        if predicted == 0:
            percentage_correctly_predicted = 0
        else:
            percentage_correctly_predicted = correctly_predicted/predicted*100

        print('| {:30s} | {:12d} | {:18.2f} | {:12d} | {:22.2f} | {:22d} | {:22.2f} |'.format(action_name, actual, percentage_actual, predicted, percentage_predicted, correctly_predicted, percentage_correctly_predicted))
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print()
    print('Correct classifications: {:d} out of {:d} possible resulting in a top-1 accuracy of {:.2f}%'.format(correct_classifications, test_size, correct_classifications/test_size*100))
    print()
    
    