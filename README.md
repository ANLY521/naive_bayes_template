# Naive Bayes for authorship attribution

# Experimental results
TODO: Add your description of this experiment here (see Homework description below)

# Files

## Lab, week 1 : `util.py`

Implements utility functions for supervised learning to be imported in `lab_nb.py` and `multinomial_nb.py`.
There is no main routine, so a helper script `test_util.py` confirms that they work.

The functions are 
* splitting data (copy-paste from your last homework?)
* creating labels as numpy arrays
* implementing the zero-rule algorithm as a baseline / scoring accuracy

Usage: `python test_util.py --path imdb_practice.txt`

## Lab, week 2 : `lab_nb.py`

This "artisinal" Naive Bayes lab implements the math of the model by hand!

Usage: `python lab_nb.py --path imdb_practice.txt`

## Homework : `multinomial_nb.py`

Usage: `python multinomial_nb.py --function_words_path ewl_function_words.txt --path imdb_practice.txt`

Apply `sklearn.naive_bayes.MultinomialNB` to two authors, as defined in the starter code. Consult the scikit learn docs to better understand how to interact with this model.

Refer to the feature extraction homework to create data in the right format for this model 
- i.e. concatenate all feature vectors to create input matrix X; create a label vector y.

Assign your data to train and test sets: 90% train and 10% test. use this same split for all experiments.


Fit and evaluate three models; our metric is accuracy:
* zero-rule baseline
* Multinomial Naive Bayes with count features
* Bernoulli Naive Bayes with binary features

Add a brief summary (~2 paragraphs) of the dataset, methods and results (i.e. model accuracy), comparing the test results on your two models and the baseline.

_For fun, you can rerun your code multiple times. 
Naive Bayes is *deterministic*, meaning the model's probability estimates are always the same, given the same inputs. 
However, but your random split should be different each time. You may see very different scores._
