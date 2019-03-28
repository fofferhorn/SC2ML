import numpy as np 
import os
import math

def load_data_with_game_crossover(data_path, train, validation, test, seed = None, maxes_path = None, normalized_data_path = None):
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
    if normalized_data_path is not None:
        data = np.load(normalized_data_path)
    else:
        min_max_norm(data, maxes_path)
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

def load_data_without_game_crossover(data_path, train, validation, test, seed = None, maxes_path = None, normalized_data_path = None):
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

    amount_train_data_points = 0
    for path in train_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                amount_train_data_points += 1

    amount_validation_data_points = 0
    for path in validation_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                amount_validation_data_points += 1

    amount_test_data_points = 0
    for path in test_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                amount_test_data_points += 1
    
    data = []
    labels = []
    for path in data_paths:
        for data_point in np.load(path):
            if len(data_point) == 248:
                data.append(data_point[:-54])
                labels.append(data_point[-54:])

    print('Normalizing data...')
    if normalized_data_path is not None:
        data = np.load(normalized_data_path)
    else:
        min_max_norm(data, maxes_path)
    print('Data normalized.')
    
    train_end = amount_train_data_points
    validation_end = amount_train_data_points + amount_validation_data_points

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
            max = 0.0
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
        norm_point = []
        for j in range(len(data[i])):
            if maxes[j] == 0.0:
                norm_point.append(0.0)
            else:
                norm_value = data[i][j]/maxes[j]
                if math.isnan(norm_value):
                    norm_point.append(0.0)
                else:
                    norm_point.append(norm_value)
        norm_data.append(norm_point)

    np.save('normalized_data.npy', norm_data)

    return norm_data


if __name__ == "__main__":
    load_data_without_game_crossover('extracted_actions', 0.7, 0.15, 0.15)

    data = np.load('normalized_data.npy')

    for i in range(len(data)):
        if len(data[i]) != 194:
            print('LENGTH ERROR AT ' + str(i) + ' ACTUAL LENGTH IS ' + str(len(data[i])))
        for j in range(len(data[i])):
            has_printed = False
            if has_printed and (0.0 > data[i][j] or data[i][j] > 1.0):
                print('RANGE ERROR IN ' + str(data[i]))
                has_printed = True