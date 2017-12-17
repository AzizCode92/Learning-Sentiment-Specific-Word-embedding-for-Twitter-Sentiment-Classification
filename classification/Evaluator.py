from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold

import copy
import csv

import numpy as np

verbose_level = 0  # verbose level
n_job = 3        # number of CPU used in evaluation
seed = 7           # seed for our random state cross validation

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
from sklearn.preprocessing import scale

class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self):
        super(Evaluator, self).__init__()

    def eval_with_test_set(self, model, feature_extractors, training_set, test_set, outfile="results_test.csv"):
        if not isinstance(feature_extractors, list):
            return

        training_contents = training_set.get_contents()
        training_labels = training_set.get_labels()

        # build training features

        # Print properties
        print "Evaluation method: Test Set"
        print "Classifier: %s" % (model.get_classifier_type())

        test_contents = test_set.get_contents()
        test_labels = test_set.get_labels()

        field_names = ["id", "content", "polarity"]
        fe_predictions = dict()

        for feature_extractor in feature_extractors:
            fe = copy.copy(feature_extractor)
            print "\nFeature Extractor: %s" % (fe.get_name())
            field_names.append(fe.get_name())

            # build our feature extractor from the training dataset contents

            fe.set_dataset(training_contents)
            fe.build()
            training_contents = [tweet.split() for tweet in training_contents]

            training_features = fe.extract_features(training_contents)
            #print("training features :")
            #print(training_features)
            # build features for our test dataset
            test_contents = [tweet.split() for tweet in test_contents]
            test_features = fe.extract_existing_features(test_contents)
            #print("test features :")
            #print(test_features)
            # build training models
            model.classify_raw(training_features, training_labels)

            # start evaluating with test set
            test_predictions = model.test_raw(test_features)
            fe_predictions[fe.get_name()] = test_predictions

            # evaluate confusion matrix
            cnf_matrix = confusion_matrix(test_labels, test_predictions,labels=['positive', 'negative','neutral'])

            print "Average F-measure: %f" % (f1_score(test_labels, test_predictions, average='macro'))
            print "Average accuracy : %f" % (f1_score(test_labels, test_predictions, average='micro'))
            print "\nConfusion Matrix:"
            print "\t\tPositive\tNegative\tNeutral (predicted labels)"
            print "Positive\t%d\t\t%d\t\t%d" % (cnf_matrix[0][0], cnf_matrix[0][1],cnf_matrix[0][2])
            print "Negative\t%d\t\t%d\t\t%d" % (cnf_matrix[1][0], cnf_matrix[1][1],cnf_matrix[1][2])
            print "Neutral \t%d\t\t%d\t\t%d" % (cnf_matrix[2][0], cnf_matrix[2][1],cnf_matrix[2][2])
            print "(actual labels)\n"

        with open(outfile, "wb") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            for i in xrange(len(test_contents)):
                row = {
                    'id': i + 1,
                    'content': test_contents[i],
                    'polarity': test_labels[i],
                }
                # append results
                for j in xrange(len(feature_extractors)):
                    row[feature_extractors[j].get_name()] = fe_predictions[feature_extractors[j].get_name()][i]

                writer.writerow(row)

    def eval_with_cross_validation(self, model, feature_extractors, training_set, num_fold=10, cv=None):
        if not isinstance(feature_extractors, list):
            return

        # if model
        training_contents = training_set.get_contents()
        training_labels = training_set.get_labels()

        # Print properties
        print "Evaluation method: Cross Validation"
        print "Number of Folds: %d" % (num_fold)
        print "Classifier: %s" % (model.get_classifier_type())

        if not cv:
            kfold = KFold(n_splits=num_fold, random_state=seed)
        else:
            kfold = cv

        for feature_extractor in feature_extractors:
            fe = copy.copy(feature_extractor)
            print "\nFeature Extractor: %s" % (fe.get_name())

            # build our feature extractor from the dataset contents
            fe.set_dataset(training_contents)
            fe.build()
            training_contents = [tweet.split() for tweet in training_contents]
            training_features = fe.extract_features(training_contents)
            # obtain our classification results
            # measure is done by using macro F1 score
            scores = cross_val_score(model.classifier, X=training_features,
                                    y=training_labels, cv=kfold, n_jobs=n_job,
                                    scoring='f1_macro', verbose=verbose_level)

            # print each of the iteration scroe
            for i in xrange(0, len(scores)):
                print "Iteration %d = %f" % (i + 1, scores[i])

            print "Average score: %f" % (scores.mean())
            print "Standard Deviation: %f" % (scores.std())
            print "Maximum F1-score: %f" % (np.amax(scores))


    def create_evaluation_result(self, model, feature_extractors, training_set, num_fold=10, outfile="results_cv.csv", cv=None):
        if not isinstance(feature_extractors, list):
            return

        # if model
        training_contents = training_set.get_contents()
        training_labels = training_set.get_labels()

        # Print properties
        print "Evaluation method: Cross Validation"
        print "Number of Folds: %d" % (num_fold)
        print "Classifier: %s" % (model.get_classifier_type())

        field_names = ["id", "content", "polarity"]
        fe_predictions = dict()

        if not cv:
            kfold = KFold(n_splits=num_fold, random_state=seed)
        else:
            kfold = cv

        for feature_extractor in feature_extractors:
            fe = copy.copy(feature_extractor)
            field_names.append(fe.get_name())

            # build our feature extractor from the dataset contents
            fe.set_dataset(training_contents)
            fe.build()
            training_contents = [tweet.split() for tweet in training_contents]
            training_features = fe.extract_features(training_contents)
            # obtain our classification results
            # measure is done by using macro F1 score
            predictions = cross_val_predict(model.classifier, X=training_features,
                                            y=training_labels, cv=kfold, n_jobs=n_job,
                                            verbose=verbose_level,fit_params={})
            fe_predictions[fe.get_name()] = predictions

        with open(outfile, "wb") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            for i in xrange(len(training_contents)):
                row = {
                    'id': i + 1,
                    'content': training_contents[i],
                    'polarity': training_labels[i],
                }
                # append results
                for j in xrange(len(feature_extractors)):
                    row[feature_extractors[j].get_name()] = fe_predictions[feature_extractors[j].get_name()][i]

                writer.writerow(row)

        return outfile
