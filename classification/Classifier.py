import sys

from sklearn import metrics

from feature_extractor import FeatureExtractor
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Classifier(object):
    """docstring for Classifier"""

    def __init__(self, models="multinomial"):
        super(Classifier, self).__init__()
        self.models = models
        if models == "multinomial":
            self.classifier = MultinomialNB()
        elif models == "svm":
            self.classifier = SVC(kernel='linear')
        elif models == "rfc":
            self.classifier = RandomForestClassifier()
        elif models == "nn":
            self.classifier = MLPClassifier()

    def classify(self, dataset):
        contents = dataset.get_contents()
        labels = dataset.get_labels()
        return self.classify_raw(contents, labels)

    def classify_raw(self, dataset, labels):
        self.classifier = self.classifier.fit(dataset, labels)
        return self.classifier

    """Return predictions for dataset using Dataset class"""

    def test(self, dataset):
        contents = dataset.get_contents()
        return self.test_raw(contents)

    """Return predictions for dataset using raw array dataset"""

    def test_raw(self, dataset):
        predictions = self.classifier.predict(dataset)
        return predictions

    def get_classifier_type(self):
        if self.models == "multinomial":
            return "Multinomial Naive-Bayes"
        elif self.models == "svm":
            return "Support Vector Machine"
        elif self.models == "rfc":
            return "Random Forest Classifier"
        elif self.models == "nn":
            return "Multilayer Perceptron (Neural Network)"
        else:
            return "Unknown classifier"


def main(filename):
    fe = FeatureExtractor("tfidf", filename)
    fe.load_dataset()
    fe.load_labels()

    bow = fe.build_bag()
    bag = fe.build_tfidf()

    print "** Using Multinomial NB Models **"

    # TFIDF
    clf = Classifier(models="multinomial")
    clf.classify(bag, fe.raw_labels)

    preds = clf.test(bag)
    # for doc, cat in zip(fe.dataset, preds):
    # 	print "%r => %s" % (doc, cat)

    print "TFIDF accuracy score: %f" % (metrics.accuracy_score(fe.raw_labels, preds, normalize=True))
    f1_pos = metrics.f1_score(fe.raw_labels, preds, pos_label='positive')
    f1_neg = metrics.f1_score(fe.raw_labels, preds, pos_label='negative')
    f1_neu = metrics.f1_score(fe.raw_labels, preds, pos_label='neutral')
    print "TFIDF F1 score: %f" % f1_pos
    print "TFIDF F1 negative score: %f" % f1_neg
    print "TFIDF F1 neutral score: %f" % f1_neu

    print "\nAverage F-measure: %f" % ((f1_pos + f1_neg + f1_neu ) / 2)

    # bag of words
    clf = Classifier(models="multinomial")
    clf.classify(bow, fe.raw_labels)
    preds = clf.test(bow)

    print "BOW accuracy score: %f" % (metrics.accuracy_score(fe.raw_labels, preds, normalize=True))
    print "BOW F1 score: %f" % (metrics.f1_score(fe.raw_labels, preds, pos_label='positive'))

    print "\n** Using SVM **"

    # TFIDF
    clf = Classifier(models="svm")
    clf.classify(bag, fe.raw_labels)

    preds = clf.test(bag)
    # for doc, cat in zip(fe.dataset, preds):
    # 	print "%r => %s" % (doc, cat)

    print "TFIDF accuracy score: %f" % (metrics.accuracy_score(fe.raw_labels, preds, normalize=True))

    # bag of words
    clf = Classifier(models="svm")
    clf.classify(bow, fe.raw_labels)
    preds = clf.test(bow)

    print "BOW accuracy score: %f" % (metrics.accuracy_score(fe.raw_labels, preds, normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(bow, fe.raw_labels, test_size=0.4, random_state=0)
    clf = Classifier(models="svm")
    clf.classify(X_train, y_train)
    preds = clf.test(X_test)

    print "Using 60/40, BOW accuracy: %f" % (metrics.accuracy_score(y_test, preds, normalize=True))
    print "Using 60/40, BOW F1: %f" % (metrics.f1_score(y_test, preds, pos_label='positive'))


if __name__ == '__main__':
    main(sys.argv[1])
