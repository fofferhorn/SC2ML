# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, optimizers, layers, losses, models, utils
from tensorflow.keras import backend as K

import os

from absl import app
from absl import flags
import sys

import math
import numpy as np

import constants as c
import utils


FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the training data.')
flags.DEFINE_string(name = 'model_name', default = 'model.h5', help = 'The name to save the model with.')
flags.DEFINE_string(name = 'maxes_path', default = None, help = 'The name of the files that contains the max values to be used for min-max normalization.')
flags.DEFINE_string(name = 'normalized_data_path', default = None, help = 'The path to the normalized data.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'The seed used to split the data.')
flags.DEFINE_integer(name = 'start_time', default = 0, help = 'The seed used to split the data.')
flags.DEFINE_integer(name = 'end_time', default = 9408, help = 'The seed used to split the data.')

FLAGS(sys.argv)
    

def test_network(test_data, test_labels):
    model = models.load_model(FLAGS.model_name, {"top_1_categorical_accuracy": utils.top_1_categorical_accuracy, "top_3_categorical_accuracy": utils.top_3_categorical_accuracy})

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
        
        percentage_actual = actual/len(test_data)*100

        predicted = 0
        correctly_predicted = 0
        for j in range(len(test_predictions)):
            if np.argmax(test_predictions[j]) == i:
                predicted += 1
            
                if np.argmax(test_labels[j]) == i:
                    correctly_predicted += 1

        percentage_predicted = predicted/len(test_data)*100
        if predicted == 0:
            percentage_correctly_predicted = 0
        else:
            percentage_correctly_predicted = correctly_predicted/predicted*100

        print('| {:30s} | {:12d} | {:18.2f} | {:12d} | {:22.2f} | {:22d} | {:22.2f} |'.format(action_name, actual, percentage_actual, predicted, percentage_predicted, correctly_predicted, percentage_correctly_predicted))
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print()
    print('Correct classifications: {:d} out of {:d} possible resulting in a top-1 accuracy of {:.2f}%'.format(correct_classifications, len(test_data), correct_classifications/len(test_data)*100))
    print()


if __name__ == "__main__":
        # _, _, _, _, test_data, test_labels = \
    #     utils.load_data_without_game_crossover(
    #         FLAGS.data_path, 
    #         0.7, 
    #         0.15, 
    #         0.15, 
    #         FLAGS.seed,
    #         FLAGS.maxes_path,
    #         FLAGS.normalized_data_path
    #     )

    _, _, _, _, test_data, test_labels = \
        utils.load_data_part_of_game(
            FLAGS.data_path, 
            0.7, 
            0.15, 
            0.15, 
            FLAGS.start_time,
            FLAGS.end_time,
            FLAGS.maxes_path,
            FLAGS.seed
        )

    test_network(test_data, test_labels)