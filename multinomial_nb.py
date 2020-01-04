#!/usr/bin/env python
import argparse

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words


def main(data_file, vocab_path):
    """Build an authorship attribution classifier using MultinomialNaiveBayes for two authors"""

    function_words = load_function_words(vocab_path)

    reviews = []
    authors = []
    with open(data_file, 'r') as data_file:
        for line in data_file:
            fields = line.strip().split("\t")
            reviews.append(fields[-1])
            authors.append(fields[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="imdb_twoauthor.tsv",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
