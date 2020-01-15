from util import split_data, labels_to_key, parse_imdb, labels_to_y, find_zero_rule_class, apply_zero_rule
import argparse

def main(data_file):
    print(data_file)

    # load the data
    authors, reviews = parse_imdb(data_file)
    num_reviews = len(reviews)
    print(f"Working with {num_reviews} reviews")

    # create a key that links author id string -> integer
    author_key = labels_to_key(authors)
    print(len(author_key))
    print(author_key)

    # convert all the labels using the key
    y = labels_to_y(authors, author_key)
    assert y.size == len(authors), f"Size of label array (y.size) must equal number of labels {len(authors)}"

    # shuffle and split the data
    train, test = split_data(reviews, y, 0.3)
    data_size_after = len(train[1]) + len(test[1])

    assert data_size_after == y.size, f"Number of datapoints after split {data_size_after} must match size before {y.size}"
    print(f"{len(train[0])} in train; {len(test[0])} in test")

    # learn zero rule on train
    train_y = train[1]
    most_frequent_class = find_zero_rule_class(train_y)

    # lookup label string from class #
    reverse_author_key = {v:k for k,v in author_key.items()}
    print(f"The most frequent class is {reverse_author_key[most_frequent_class]}")

    # apply zero rule to test reviews
    test_predictions = apply_zero_rule(test[0], most_frequent_class)
    print(f"Zero rule predictions on held-out data: {test_predictions}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test supervised learning utilities')
    parser.add_argument('--path', type=str, default="imdb_twoauthor.tsv",
                        help='path to author dataset')
    args = parser.parse_args()

    main(args.path)