import os
from shutil import copyfile

from absl import app
from absl import flags
import sys


import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the data.')
flags.DEFINE_string(name = 'save_path', default = 'split', help = 'Where to save the data split.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'A seed for the random split.')

FLAGS(sys.argv)


def split(data_path, save_path, train, validation, test, seed):
    if train + validation + test != 1:
        raise ValueError('train: ' + str(train) + ', validation: ' + str(validation) + ' and test: ' + str(test) + ' must sum to 1')

    # Check if the replays_path exists
    if not os.path.isdir(data_path):
        raise ValueError('The path ' + data_path + ' does not exist.')

    # Make list of the paths to all the replays
    cwd = os.getcwd()
    data_paths = []
    for data in os.listdir(data_path):
        data_path = os.path.join(cwd, data_path, data)
        if os.path.isfile(data_path) and data.lower().endswith('.sc2replay'):
            data_paths.append(data_path)

    # Check if the save_path exists. Otherwise we need to create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(data_paths)

    train_end = int(len(data_paths) * train)
    validation_end = int(len(data_paths) * (train + validation))

    for path in data_paths[:train_end]:
        copyfile(path, os.path.join(cwd, save_path, 'train', path.split('/')[-1]))

    for path in data_paths[train_end: validation_end]:
        copyfile(path, os.path.join(cwd, save_path, 'validation', path.split('/')[-1]))

    for path in data_paths[validation_end:]:
        copyfile(path, os.path.join(cwd, save_path, 'test', path.split('/')[-1]))


def main(argv):
    split(FLAGS.data_path, FLAGS.save_path, 0.7, 0.2, 0.1, FLAGS.seed)


if __name__ == "__main__":
    app.run(main)