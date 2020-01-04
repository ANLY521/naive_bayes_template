# Naive Bayes for authorship attribution

# Experimental results
TODO: Add your description of this experiment here (see Homework description below)

# Files

## Lab : `lab_nb.py`

This "artisinal" Naive Bayes lab implements the math of the model by hand!

Usage: `python lab_nb.py --path imdb_practice.txt`

## Homework : `multinomial_nb.py`

Usage: `python multinomial_nb.py --function_words_path ewl_function_words.txt --path imdb_practice.txt`

Apply `sklearn.naive_bayes.MultinomialNB` to two authors, as defined in the starter code. Consult the scikit learn docs to better understand how to interact with this model.

Use last week’s code to create data in the right format for this model - i.e. concatenate all feature vectors to create input matrix X; create a label vector y.

Split your data in 90% train and 10% test

Determine the most common class baseline on test. (The "zero rule algorithm" described here is a good way to get the baseline: https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/ (Links to an external site.))

Using the same data split for both, fit one model using count features and a second model using binary features.

Add a brief summary (~2 paragraphs) of the dataset, methods and results (i.e. model accuracy), comparing the test results on your two models and the baseline.

_For fun, you can rerun your code multiple times. The model's probability estimates are always the same, given the same inputs, but your random split should be different each time. You may see very different scores._
