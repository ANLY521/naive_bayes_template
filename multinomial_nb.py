#!/usr/bin/env python
import argparse
from util import load_function_words, parse_imdb


def main(data_file, vocab_path):
    """Build and evaluate an authorship attribution classifier using MultinomialNaiveBayes for two authors"""

    function_words = load_function_words(vocab_path)

    reviews, authors = parse_imdb(data_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="imdb_twoauthor.tsv",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
