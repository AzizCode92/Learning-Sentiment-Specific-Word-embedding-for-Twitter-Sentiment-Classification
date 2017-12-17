#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import logging
import numpy as np
from ConfigParser import ConfigParser
from itertools import chain
# local
from deepnl import *
from deepnl.extractors import *
from deepnl.reader import TweetReader
from deepnl.network import Network
from deepnl.sentiwords import SentimentTrainer


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class sswe_model(object):
    def __init__(self, window=3, embeddings_size=50, epochs=100, learning_rate=0.001,
                 eps=1e-8, ro=0.95, hidden=200, ngrams=2, textField=0,
                 tagField=1, alpha=0.5, train=None, model=None,
                 vocab=None, minOccurr=3, vocab_size=0, vectors=None, load=None,
                 threads=5, variant=None, verbose=None, config_file=None):
        self.window = window
        self.embeddings_size = embeddings_size
        self.iterations = epochs
        self.learning_rate = learning_rate
        self.eps = eps
        self.ro = ro
        self.hidden = hidden
        self.ngrams = ngrams
        self.textField = textField
        self.tagField = tagField
        self.alpha = alpha
        self.train = train
        self.vocab = vocab
        self.minOccurr = minOccurr
        self.vocab_size = vocab_size
        self.vectors = vectors
        self.load = load
        self.variant = variant
        self.verbose = verbose
        self.model = model
        self.threads = threads
        self.config_file = config_file



def create_sswe_model(train_filename, vocab_file, vector_file, train_model, save_model, size):
    """model parameters: you can customize other parameters in the class sswe_mode()"""
    emb_size = size  # Number of features per word
    epochs = 100  # Number of training epochs
    l_r = 0.1  # Learning rate for network weights
    hidden = 200   # Number of hidden neurons
    ngrams = 2  # Length of ngrams
    text = 0  # field containing text
    tag = 1  # field containing polarity
    train = train_filename  # File with text corpus for training
    model = save_model  # File where to save the model
    vocab = vocab_file  # Vocabulary file, either read and updated or created
    vectors = vector_file  # Embeddings file, either read and updated or created
    load = train_model  # Load previously saved model
    threads = 15  # Number of threads
    variant = None

    sswe = sswe_model(embeddings_size=emb_size, epochs=epochs, learning_rate=l_r, threads=threads,
                      hidden=hidden, ngrams=ngrams, textField=text, tagField=tag, train=train,
                      model=model, vocab=vocab, minOccurr=3, vectors=vectors, load=load, variant=variant)
    return sswe



def sswe_trainer(model_parameters):
    # set the seed for replicability
    np.random.seed(42)
    # args = parser.parse_args()
    args = model_parameters
    log_format = '%(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_level = logging.INFO
    logging.basicConfig(format=log_format, level=log_level)
    logger = logging.getLogger("Logger")

    config = ConfigParser()
    if args.config_file:
        config.read(args.config_file)
    # merge args with config
    reader = TweetReader(text_field=args.textField, label_field=args.tagField, ngrams=args.ngrams)
    reader.read(args.train)
    vocab, bigrams, trigrams = reader.create_vocabulary(reader.sentences, args.vocab_size,
                                                        min_occurrences=args.minOccurr)
    #print("length vocab")
    #print(len(vocab))
    if args.variant == 'word2vec' and os.path.exists(args.vectors):
        embeddings = Embeddings(vectors=args.vectors, variant=args.variant)
        embeddings.merge(vocab)
        logger.info("Saving vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)
    elif os.path.exists(args.vocab):
        # start with the given vocabulary
        b_vocab = reader.load_vocabulary(args.vocab)
        bound = len(b_vocab)-len(bigrams)-len(trigrams)
        base_vocab=b_vocab[:bound]
        #print("length base vocab :")
        #print(len(base_vocab))
        if os.path.exists(args.vectors):
            # load embeddings
            embeddings = Embeddings(vectors=args.vectors, vocab=base_vocab, variant=args.variant)
        else:
            # create embeddings
            embeddings = Embeddings(args.embeddings_size, vocab=base_vocab, variant=args.variant)
            # add the ngrams from the corpus
            embeddings.merge(vocab)
            logger.info("Overriding vocabulary in %s" % args.vocab)
            embeddings.save_vocabulary(args.vocab)
    else:
        embeddings = Embeddings(args.embeddings_size, vocab=vocab, variant=args.variant)
        logger.info("Saving vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)

    # Assume bigrams are prefix of trigrams, or else we should put a terminator
    # on trie
    trie = {}
    for b in chain(bigrams, trigrams):
        tmp = trie
        for w in b:
            tmp = tmp.setdefault(embeddings.dict[w], {})

    converter = Converter()
    converter.add(embeddings)

    trainer = create_trainer(args, converter)

    report_intervals = max(args.iterations / 200, 1)
    report_intervals = 10000  # DEBUG

    logger.info("Starting training")

    # a generator expression (can be iterated several times)
    # It caches converted sentences, avoiding repeated conversions
    converted_sentences = converter.generator(reader.sentences, cache=True)
    trainer.train(converted_sentences, reader.polarities, trie,
                  args.iterations, report_intervals)

    logger.info("Overriding vectors to %s" % args.vectors)
    embeddings.save_vectors(args.vectors, args.variant)
    if args.model:
        logger.info("Saving trained model to %s" % args.model)
        trainer.save(args.model)


def create_trainer(args, converter):
    """
    Creates or loads a neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = SentimentTrainer.load(args.load)
        # change learning rate
        trainer.learning_rate = args.learning_rate
    else:
        logger.info('Creating new network...')
        # sum the number of features in all extractors' tables
        input_size = converter.size() * (args.window * 2 + 1)
        nn = Network(input_size, args.hidden, 2)
        options = {
            'learning_rate': args.learning_rate,
            'eps': args.eps,
            'ro': args.ro,
            'verbose': args.verbose,
            'left_context': args.window,
            'right_context': args.window,
            'ngram_size': args.ngrams,
            'alpha': args.alpha
        }
        trainer = SentimentTrainer(nn, converter, options)

    trainer.saver = saver(args.model, args.vectors)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())

    return trainer


def saver(model_file, vectors_file):
    """Function for saving model periodically"""

    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.save_vectors(vectors_file)
        if model_file:
            trainer.save(model_file)

    return save


def buildWordVector(tokens, size, tweet_w2v, tfidf):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


def get_sswe_features(vocab_file, model_file):
    vocabs = []
    models = []
    with open(vocab_file, "rb") as vocablist:
        for vocab in vocablist:
            vocabs.append(vocab.rstrip())

    with open(model_file, "rb") as modellist:
        for model in modellist:
            arr_model = model.split()
            models.append(np.array(map(float, arr_model)))
    # build our word embedding model vectorizer
    sswe_dict = dict(zip(vocabs, models))
    return sswe_dict
