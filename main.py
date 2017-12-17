from preprocessing.preprocesstweets import *
from embedding.sswe_extractor import *
from models import Dataset
from classification.Classifier import Classifier
from classification.NeuralNets import *
from classification.Evaluator import Evaluator
from feature_extracting import SennaFeatureExtractor
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from tqdm import tqdm
import numpy as np


def main():
    """ Sentiment Specific Embedding for twitter classification """

    embeddings_size = 50  # Embedding size for SSWE model
    vocab_file = "Embedding/features/semeval_vocabs_200.txt"  # path to the vocabulary file
    vector_file = "Embedding/features/semeval_vectors_200.txt"  # path to the vector file
    stopwordsfile = "preprocess/stopwords.txt"

    """     Sentiment-Specific Word Embedding (SSWE)    """

    if True:
        # Load dataset
        data_train = 'dataset/training1600000.csv'  # training data set file path
        pre_data_train = 'dataset/preprocessed_dataset1600000.csv'  # file to save dataset after cleaning

        if True:
            print("\n **** Dataset cleaning ****")
            tweets_prepocess(data_train, pre_data_train, stopwordsfile)

        if True:
            print("\n **** SSWE model Trainig ****")
            train_model = None  # path to the file contains the trained model if it is already exist
            save_model = "Embedding/models/SSWE_model_1600000_200"  # path to the file where model will be saved
            sswe = create_sswe_model(pre_data_train, vocab_file, vector_file, train_model,
                                     save_model, embeddings_size)
            sswe_trainer(sswe)

    """     Embedding visualisation and Similarity computing    """

    if True:
        visualiser = Visualiser(sizeOfEmbedding=embeddings_size,
                                VocabsFname=vocab_file,
                                VectorsFname=vector_file,
                                WVFilename="Visualisation/data/w2vformat.txt",
                                visualizerHTMLfilename="Visualisation/data/embedding.html")
        visualiser.visualize()

    """ Twitter Sentiment Classification """

    if True:
        # Data pre-processing

        print("\n **** Training data cleaning ****")
        pre_processing_train = "dataset/preprocessed_semeval_traindataset.csv"
        # tweets_prepocess(train_set, pre_processing_train, stopwordsfile)

        print("\n **** Test data cleaning ****")
        pre_processing_test = "dataset/preprocessed_semeval_testdataset.csv"
        # tweets_prepocess(test_set, pre_processing_test, stopwordsfile)

        # LOAD TRAIN SET
        dataset_train = Dataset.DatasetReview()
        dataset_train.load_review_from_csv(pre_processing_train)

        # LOAD TEST SET
        dataset_test = Dataset.DatasetReview()
        dataset_test.load_review_from_csv(pre_processing_test)

        ################################### Neural Nets classifier ###########################

        # Extract Features
        tweet2v = get_sswe_features(vocab_file, vector_file)

        # Extract samples and labels
        x_train, y_train = split_data(dataset_train)
        x_test, y_test = split_data(dataset_train)

        tfidf = build_tfidf(x_train)

        train_vecs_sswe = np.concatenate(
            [buildWordVector(z.split(), embeddings_size,
                             tweet2v, tfidf) for z in tqdm(map(lambda x: x, x_train))])

        train_vecs_sswe = scale(train_vecs_sswe)

        test_vecs_sswe = np.concatenate(
            [buildWordVector(z.split(), embeddings_size,
                             tweet2v, tfidf) for z in tqdm(map(lambda x: x, x_test))])
        test_vecs_sswe = scale(test_vecs_sswe)

        # neural network model
        neuralnets = NeuralNets(input_size=embeddings_size, x_train=train_vecs_sswe, y_train=y_train,
                                epochs=450, batch_size=32, x_test=test_vecs_sswe, y_test=y_test)
        neuralnets.train_neural_nets()

        ##########################################################################################
        ########
        ########        Classical classifiers with sklearn
        ########
        ##########################################################################################
        print("\n**** CROSS VALIDATION EVALUATION (CORPUS: SemEval) ****\n")

        fe_sswe = SennaFeatureExtractor(infile=vector_file, vocabfile=vocab_file, dimen=embeddings_size)
        feature_extractors = [fe_sswe]
        ev = Evaluator()

        ################################# SVM ###################################################

        print ("\n**** CROSS VALIDATION EVALUATION (CORPUS: SemEval) ****\n")
        model = Classifier(models="svm")
        kfold = KFold(n_splits=10)
        ev.eval_with_cross_validation(model, feature_extractors=feature_extractors,
                                      training_set=dataset_train, num_fold=10, cv=kfold)
        ev.create_evaluation_result(model, feature_extractors=feature_extractors,
                                    training_set=dataset_train, num_fold=10, cv=kfold)

        print ("\n**** TEST SET EVALUATION (CORPUS: SemEval) ****\n")
        ev.eval_with_test_set(model, feature_extractors=feature_extractors,
                              training_set=dataset_train,
                              test_set=dataset_test)

        ################################### Naive bayes ##########################################

        print ("\n**** CROSS VALIDATION EVALUATION (CORPUS: SemEval) ****\n")
        model = Classifier(models="multinomial")
        kfold = KFold(n_splits=10)
        ev.eval_with_cross_validation(model, feature_extractors=feature_extractors,
                                      training_set=dataset_train, num_fold=10, cv=kfold)
        ev.create_evaluation_result(model, feature_extractors=feature_extractors,
                                    training_set=dataset_train, num_fold=10, cv=kfold)

        print ("\n**** TEST SET EVALUATION (CORPUS: DATASET) ****\n")
        ev.eval_with_test_set(model, feature_extractors=feature_extractors,
                              training_set=dataset_train,
                              test_set=dataset_test)

        #########################################  RandomForestClassifier #######################

        print ("\n**** CROSS VALIDATION EVALUATION (CORPUS: SemEval) ****\n")
        model = Classifier(models="rfc")
        kfold = KFold(n_splits=10)
        ev.eval_with_cross_validation(model, feature_extractors=feature_extractors,
                                      training_set=dataset_train, num_fold=10, cv=kfold)
        ev.create_evaluation_result(model, feature_extractors=feature_extractors,
                                    training_set=dataset_train, num_fold=10, cv=kfold)

        print ("\n**** TEST SET EVALUATION (CORPUS: SemEval) ****\n")
        ev.eval_with_test_set(model, feature_extractors=feature_extractors,
                              training_set=dataset_train,
                              test_set=dataset_test)

        #########################################  MLPClassifier #######################

        print ("\n**** CROSS VALIDATION EVALUATION (CORPUS: SemEval) ****\n")
        model = Classifier(models="nn")
        kfold = KFold(n_splits=10)
        ev.eval_with_cross_validation(model, feature_extractors=feature_extractors,
                                      training_set=dataset_train, num_fold=10, cv=kfold)
        ev.create_evaluation_result(model, feature_extractors=feature_extractors,
                                    training_set=dataset_train, num_fold=10, cv=kfold)

        print ("\n**** TEST SET EVALUATION (CORPUS: SemEval) ****\n")
        ev.eval_with_test_set(model, feature_extractors=feature_extractors,
                              training_set=dataset_train,
                              test_set=dataset_test)


if __name__ == '__main__':
    main()
