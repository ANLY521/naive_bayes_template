import numpy as np
from collections import Counter

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words

def parse_imdb(data_file):
    authors = []
    reviews = []
    with open(data_file, 'r') as data_file:
        for line in data_file:
            fields = line.strip().split("\t")
            reviews.append(fields[-1])
            authors.append(fields[0])
    return authors, reviews


def split_data(X, y, test_percent = 0.3, shuffle=True):
    """

    :param X:
    :param y:
    :param split_percent:
    :return:
    """
    if shuffle:
        print('shuffle not implemented')
    data_size = len(X)
    num_test = int(test_percent * data_size)

    train = (X[:-num_test], y[:-num_test])
    test = (X[-num_test:], y[-num_test:])
    return train, test


def labels_to_key(labels):
    """

    :param labels:
    :return:
    """
    label_set = set(labels)
    label_key = {}
    for i, label in enumerate(label_set):
        label_key[label] = i
    return label_key


def labels_to_y(labels, label_key):
    y = np.zeros(len(labels), dtype=np.int)
    for i,l in enumerate(labels):
        y[i] = label_key[l]
    return y


def find_zero_rule_class(train_y):
    class_counts = Counter(train_y)
    print(class_counts)
    most_freq = max(class_counts, key=lambda k: class_counts[k])
    return most_freq


def apply_zero_rule(y, zero_class):
    classifications = np.zeros(len(y), dtype=np.int)
    # assign every y the zero class
    classifications[:] = zero_class
    return classifications
