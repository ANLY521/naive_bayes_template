#!/usr/bin/env python
import argparse
from collections import defaultdict
from util import parse_imdb


def word_probs(list_of_reviews, feature_list):
    """calculates probabilities of each feature given this dataset using Laplace smoothing
    returns a dict {feature_1: probability_1, ... feature_n: probability_n}"""
    return {}


def score(review, author_prob, feature_probs):
    """Calculates a naive bayes score for a string, given class estimate and feature estimates"""
    tokenized_review = review.strip().lower().split()
    p = 0
    return p

def main(data_file, authors, features):
    """extract function word features from a text file"""

    # set up: create a dictionary from author -> list of reviews for two authors we will model
    two_author_reviews = defaultdict(list)

    with open(data_file, 'r') as df:
        for line in df:
            fields = line.strip().split("\t")
            author = fields[1]
            if author in authors:
                two_author_reviews[author].append(fields[-1])

    # hold out one review per author to test the model
    training_reviews = {author: reviews[:-1] for author, reviews in two_author_reviews.items()}
    heldout_reviews = {author: reviews[-1] for author, reviews in two_author_reviews.items()}

    # estimate author probabilities. Creates a dict {author_1: probability_1, ...}
    author_probs = {}

    author_word_probs = {author: word_probs(reviews, features) for author, reviews in training_reviews.items()}
    print(author_word_probs)

    # print(score(heldout_reviews['2093818'], author_probs['2093818'], author_word_probs['2093818']))
    # print(score(heldout_reviews['2093818'], author_probs['7743887'], author_word_probs['7743887']))
    # print()
    # print(score(heldout_reviews['7743887'], author_probs['2093818'], author_word_probs['2093818']))
    # print(score(heldout_reviews['7743887'], author_probs['7743887'], author_word_probs['7743887']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='artisinal naive bayes lab')
    parser.add_argument('--path', type=str, default="imdb_practice.txt",
                        help='path to author data')

    args = parser.parse_args()
    authors = ['2093818', '7743887']
    features = ["in", "while", "until", "which", "how"]

    main(args.path, authors, features)
