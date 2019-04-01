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
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    correct_classifications = 0

    for i in range(len(test_predictions)):
        prediction_action = c.protoss_macro_actions[np.argmax(test_predictions[i])]

        actual_action = c.protoss_macro_actions[np.argmax(test_labels[i])]
        
        if prediction_action == actual_action:
            correct_classifications += 1

        # print('_____________________________________________________________________________________')
        # print('{:17s} - {:3f} - {:20s}'.format('Max prediction ', prediction, prediction_action))
        # print('{:17s} - {:3f} - {:20s}'.format('Max actual ', actual, actual_action))
        # print('_____________________________________________________________________________________')

    # result = 'Action;# actual;Actual % of total;# predicted;Predicted % of total;# correctly predicted;% correctly predicted\n'
    
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

        # result += str(action_name) + ';' + str(actual) + ';' + str(percentage_actual) + ';' + str(predicted) + ';' + str(percentage_predicted) + ';' + str(correctly_predicted) + ';' + str(percentage_correctly_predicted) + '\n'

        print('| {:30s} | {:12d} | {:18.2f} | {:12d} | {:22.2f} | {:22d} | {:22.2f} |'.format(action_name, actual, percentage_actual, predicted, percentage_predicted, correctly_predicted, percentage_correctly_predicted))
    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')

    # with open('guesses.csv', 'a') as f:
    #     f.write(result)

    print()
    print('Correct classifications: {:d} out of {:d} possible resulting in a top-1 accuracy of {:.2f}%'.format(correct_classifications, len(test_data), correct_classifications/len(test_data)*100))
    print()


def test_network_intervals(intervals, seed, data_path):
    np.random.seed(seed)

    model = models.load_model(FLAGS.model_name, {"top_1_categorical_accuracy": utils.top_1_categorical_accuracy, "top_3_categorical_accuracy": utils.top_3_categorical_accuracy})

    data_paths = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path) and file.lower().endswith('.npy'):
            data_paths.append(file_path)

    np.random.shuffle(data_paths)

    test_begin = int(len(data_paths) * (0.70 + 0.15))

    test_paths = data_paths[test_begin:]

    data = []
    labels = []
    for path in test_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                data.append(data_point[:-54])
                labels.append(data_point[-54:])

    # with open('intervals.csv', 'w+') as f:
    #     f.write('total_games_amount;total_data_amount\n')
    #     f.write(str(len(test_paths)) + ';' + str(len(data)) + '\n\n')
    #     f.write('interval_start;interval_end;data_amount;loss;top_1_accuracy;top_3_accuracy\n')

    prev = 0

    for interval_end in intervals:

        interval_data = []
        interval_labels = []
        for i in range(len(data)):
            if prev <= data[i][0] < interval_end:
                interval_data.append(data[i])
                interval_labels.append(labels[i])

        if len(interval_data) == 0 or len(interval_labels) == 0:
            results = str(prev)
            results += ';' + str(interval_end)
            results += ';' + str(len(interval_data))
            results += ';;'
        else:
            interval_data = keras.utils.normalize(interval_data, axis=-1, order=2)

            scores = model.evaluate(np.array(interval_data), np.array(interval_labels), verbose=0)

            results = str(prev)
            results += ';' + str(interval_end)
            results += ';' + str(len(interval_data))
            for index in range(len(model.metrics_names)):
                results += ';%.4f' % (scores[index])

        results += '\n'

        print(results)

        with open('intervals.csv', 'a') as f:
            f.write(results)

        prev = interval_end


def random_test(data_path, seed = None):
    data_paths = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path) and file.lower().endswith('.npy'):
            data_paths.append(file_path)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(data_paths)

    test_begin = int(len(data_paths) * (0.70 + 0.15))

    test_paths = data_paths[test_begin:]

    labels = []
    for path in test_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                labels.append(data_point[-54:])

    header = 'top_1_accuracy;top_3_accuracy\n'
    print(header)
    with open('results.csv', 'w+') as f:
        f.write(header)

    for _ in range(20):
        correct_top_1_classifications = 0
        correct_top_3_classifications = 0

        for i in range(len(labels)):
            actual_label = np.argmax(labels[i])

            prediction = np.random.uniform(low=0.0, high=1.0, size=(54,))
            
            predicted_label = np.argmax(prediction)

            if predicted_label == actual_label:
                correct_top_1_classifications += 1

            predicted_labels = np.argsort(prediction)[-3:]
            
            if actual_label in predicted_labels:
                correct_top_3_classifications += 1
            
        top_1_accuracy = correct_top_1_classifications/len(labels)
        top_3_accuracy = correct_top_3_classifications/len(labels)

        results = '%.4f' % (top_1_accuracy)
        results += ';%.4f' % (top_3_accuracy)
        results += '\n'

        print(results)
        with open('results.csv', 'a') as f:
            f.write(results)


def probe_test(data_path, seed = None):
    data_paths = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path) and file.lower().endswith('.npy'):
            data_paths.append(file_path)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(data_paths)

    test_begin = int(len(data_paths) * (0.70 + 0.15))

    test_paths = data_paths[test_begin:]

    labels = []
    for path in test_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                labels.append(data_point[-54:])

    header = 'top_1_accuracy;top_3_accuracy\n'
    print(header)
    with open('results.csv', 'w+') as f:
        f.write(header)

    correct_top_1_classifications = 0
    correct_top_3_classifications = 0

    for i in range(len(labels)):
        actual_label = np.argmax(labels[i])

        predicted_label = 47 # Probe

        if predicted_label == actual_label:
            correct_top_1_classifications += 1
            correct_top_3_classifications += 1

    top_1_accuracy = correct_top_1_classifications/len(labels)
    top_3_accuracy = correct_top_3_classifications/len(labels)

    results = '%.4f' % (top_1_accuracy)
    results += ';%.4f' % (top_3_accuracy)
    results += '\n'

    print(results)
    with open('results.csv', 'a') as f:
        f.write(results)


if __name__ == "__main__":
    # for d in */ ; do cat "$d/results.txt" | grep top; done

    test_data, test_labels, _, _, _, _ = \
        utils.load_data_without_game_crossover(
            FLAGS.data_path, 
            0.7, 
            0.15, 
            0.15, 
            FLAGS.seed,
            FLAGS.maxes_path,
            FLAGS.normalized_data_path
        )

    test_network(test_data, test_labels)

    # random_test(FLAGS.data_path, FLAGS.seed)

    # probe_test(FLAGS.data_path, FLAGS.seed)

    # intervals = [672, 1344, 2016, 2688, 3360, 4032, 4704, 5376, 6048, 6720, 7392, 8064, 8736, 9408, 10080, 10752, 11424, 12096, 12768, 13440, 14112, 14784, 15456, 16128, 16800, 17472, 18144, 18816, 19488, 20160, 20832, 21504, 22176, 22848, 23520, 24192, 24864, 25536, 26208, 26880, 27552, 28224, 28896, 29568, 30240, 30912, 31584, 32256, 32928, 33600, 34272, 34944, 35616, 36288, 36960, 37632, 38304, 38976, 39648, 40320, 40992, 41664, 42336, 43008, 43680, 44352, 45024, 45696, 46368, 47040, 47712, 48384, 49056, 49728, 50400, 51072, 51744, 52416, 53088, 53760, 54432, 55104, 55776, 56448, 57120, 57792, 58464, 59136, 59808, 60480, 61152, 61824, 62496, 63168, 63840, 64512, 65184, 65856, 66528, 67200, 67872, 68544, 69216, 69888, 70560, 71232, 71904, 72576, 73248, 73920, 74592, 75264, 75936, 76608, 77280, 77952, 78624, 79296, 79968, 80640, 81312, 81984, 82656, 83328, 84000, 84672, 85344, 86016, 86688, 87360, 88032, 88704, 89376, 90048, 90720, 91392, 92064, 92736, 93408, 94080, 94752, 95424, 96096, 96768, 97440, 98112, 98784, 99456, 100128, 100800, 101472, 102144, 102816, 103488, 104160, 104832, 105504, 106176, 106848, 107520, 108192, 108864, 109536, 110208, 110880, 111552, 112224, 112896, 113568, 114240, 114912, 115584, 116256, 116928, 117600, 118272, 118944, 119616, 120288, 120960]

    # test_network_intervals(intervals, FLAGS.seed, FLAGS.data_path)