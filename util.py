import numpy as np
from collections import Counter
import json

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words

# TODO: lab 1
def parse_federalist_papers(data_file):
    authors = []
    texts = []
    essay_ids = []
    return authors, texts, essay_ids

# TODO: write this function (lab1, homework)
def labels_to_key(labels):
    """
    Creates a mapping from string representations of labels to integers
    :param labels:
    :return: label_key, dict {str: int}
    """
    label_key = {}
    return label_key

# TODO: write this function (lab1, homework)
def labels_to_y(labels, label_key):
    """
    :param labels: list of strings
    :param label_key: dictionary {str: int}
    :return: numpy vector y
    """
    y = np.zeros(len(labels), dtype=np.int)
    return y

# TODO: write this function (lab1, homework)
def find_zero_rule_class(train_y):
    """
    Determines the class predicted by the zero rule algorithm
    :param train_y: training labels
    :return: most_freq, the most frequent element in train_y
    """
    most_freq = None
    return most_freq

# TODO: write this function (lab1, homework)
def apply_zero_rule(X, zero_class):
    """
    Predicts most frequent class using zero rule algorithm
    :param X: iterable, data to classify
    :param zero_class: class to predict
    :return: classifications: numpy array
    """
    classifications = np.zeros(len(X), dtype=np.int)
    return classifications
