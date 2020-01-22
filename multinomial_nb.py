#!/usr/bin/env python
import argparse
from util import load_function_words, parse_federalist_papers, labels_to_key, labels_to_y, split_data
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# TODO create a function that loads all the essays into a matrix
def load_features(list_of_essays, list_of_features):
    total_essays = len(list_of_essays)
    X = np.zeros((total_essays, len(list_of_features)), dtype=np.int)
    # row is which essay
    for i,essay in enumerate(list_of_essays):
        review_word_counts = Counter(essay.lower().split())
        # column is which word
        for j,f_word in enumerate(list_of_features):
            if f_word in review_word_counts:
                X[i,j] = review_word_counts[f_word]
    return X


def main(data_file, vocab_path):
    """Build and evaluate Naive Bayes classifiers for the federalist papers"""

    function_words = load_function_words(vocab_path)

    authors, essays, essay_ids = parse_federalist_papers(data_file)

    function_words = load_function_words(vocab_path)

    # TODO: load the attributed essays into a feature matrix
    X = load_features(essays, function_words)
    # TODO: load the author names into a vector y, mapped to 0 and 1, using functions from util.py
    labels_map = labels_to_key(authors)
    y = np.asarray(labels_to_y(authors, labels_map))

    print(f"Numpy array has shape {X.shape} and dtype {X.dtype}")

    # TODO shuffle, then split the data
    train, val = split_data(X, y, 0.25)
    train_X,train_y = train
    print((train_y==0).sum(), (train_y==1).sum(), train_y.shape[0])
    val_X, val_y = val
    print((val_y==0).sum(), (val_y==1).sum(), val_y.shape[0])


    # TODO: train a multinomial NB model, evaluate on validation split
    mnb = MultinomialNB()
    mnb.fit(train_X, train_y)
    print("fit nb: counts")
    print(mnb.score(val_X, val_y))

    # TODO: train a Bernoulli NB model, evaluate on validation split
    bnb = BernoulliNB()

    bnb.fit(train_X>0, train_y)
    print("fit bnb: binary")
    print(bnb.score(val_X>0, val_y))

    # TODO: fit the zero rule
    most_common_class = np.argmax([(train_y==0).sum(), (train_y==1).sum()])
    print(f"most common class: {most_common_class}")

    print(f"baseline:{(val_y==most_common_class).sum()/val_y.size:.02}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="federalist_dev.json",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
