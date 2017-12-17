from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix


def split_data(dataset):

    x_samples = dataset.get_contents()
    y_labels = dataset.get_labels()
    y = []
    for x in y_labels:
        if x=="positive":
            y.append(1)
        elif x=="negative":
            y.append(-1)
        else:
            y.append(0)

    return x_samples,y


def build_tfidf(x_train):
    print 'building tf-idf matrix ...'
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    vectorizer.fit_transform([x.split() for x in x_train])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print 'vocab size :', len(tfidf)
    return tfidf


class NeuralNets(object):
    def __init__(self, input_size=100,x_train=None, y_train=None,
                 epochs=20, batch_size=32,x_test=None, y_test=None):
            self.inputdim = input_size
            self.xtrain = x_train
            self.ytrain = y_train
            self.epochs = epochs
            self.xtest = x_test
            self.ytest = y_test
            self.batchsize = batch_size


    def train_neural_nets(self):

        "*** Train Neural Networks model ***"

        model = Sequential()
        model.add(Dense(200, activation='relu', input_dim=self.inputdim))
        #model.add(Dense(32, activation='softsign'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy','mse','mae'])
        print("\n Training Neural Network Classifier with Training dataset")
        model.fit(self.xtrain, self.ytrain, epochs=self.epochs, batch_size=self.batchsize, verbose=2)
        print("\n Evaluating Neural Network Classifier on Test dataset")
        score = model.evaluate(self.xtest, self.ytest, batch_size=128, verbose=2)
        print("{} is {}".format("accuracy",score[1]))
        print("{} is {}".format("mse: ",score[2]))
        print("{} is {}".format("mae: ", score[3]))
        y_predictions = model.predict(self.xtest, batch_size=128, verbose=2)
        y_pred = np.around(y_predictions)
        y_pred = [int(x) for x in y_pred.flatten().tolist()]
        cnf_matrix = confusion_matrix(self.ytest, y_pred,labels=[1,-1,0])
        print "Average F-measure: %f" % (f1_score(self.ytest, y_pred, average='macro'))
        print "\n Confusion Matrix:"
        print "\t\tPositive\tNegative\tNeutral (predicted labels)"
        print "Positive\t%d\t\t%d\t\t%d" % (cnf_matrix[0][0], cnf_matrix[0][1], cnf_matrix[0][2])
        print "Negative\t%d\t\t%d\t\t%d" % (cnf_matrix[1][0], cnf_matrix[1][1], cnf_matrix[1][2])
        print "Neutral \t%d\t\t%d\t\t%d" % (cnf_matrix[2][0], cnf_matrix[2][1], cnf_matrix[2][2])
        print "(actual labels)\n"
