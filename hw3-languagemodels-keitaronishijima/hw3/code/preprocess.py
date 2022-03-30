import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the
    sentences to build the train and test data separately. Create a vocabulary
    dictionary that maps all the unique tokens from your train and test data as
    keys to a unique integer value. Then vectorize your train and test data based
    on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id
    form), test (1-d list or array with testing words in vectorized/id form),
    vocabulary (dict containing word->index mapping)
    """
    # TODO: load and concatenate training data from training file.
    with open(train_file) as f:
        train_data = f.readlines()
    for i in range(len(train_data)):
        train_data[i] = train_data[i].split()

    # TODO: load and concatenate testing data from testing file.
    with open(test_file) as f:
        test_data = f.readlines()
    for i in range(len(test_data)):
        test_data[i] = test_data[i].split()
    # TODO: read in and tokenize training data
    
    # Make a dictionary mapping from vocabulary to index(unique integer)
    vocab_dict = {}
    idx = 0
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            if train_data[i][j] not in vocab_dict:
                vocab_dict[train_data[i][j]] = idx
                idx += 1
    ret_train = train_data.copy()
    ret_test = test_data.copy()
    # Tokenize the train data
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            ret_train[i][j] = vocab_dict[train_data[i][j]]
    # TODO: read in and tokenize testing data
    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            ret_test[i][j] = vocab_dict[test_data[i][j]]
    # TODO: return training tokens, testing tokens, and the vocab dictionary.
    ret_train = [item for sublist in ret_train for item in sublist]
    ret_test = [item for sublist in ret_test for item in sublist]
    return ret_train, ret_test, vocab_dict
