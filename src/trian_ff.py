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

flags.DEFINE_string(name = 'data_path', default = 'extracted_actions', help = 'The path to the data.')
flags.DEFINE_string(name = 'save_path', default = 'split', help = 'Where to save the data split.')
flags.DEFINE_integer(name = 'seed', default = None, help = 'A seed for the random split.')

FLAGS(sys.argv)


def train():
    


def main():
    pass


if __name__ == "__main__":
    main()