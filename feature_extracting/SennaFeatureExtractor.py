import numpy as np

from FeatureExtractor import FeatureExtractor
from vectorizer.WordEmbeddingVectorizer import WordEmbeddingVectorizer

from models.SentenceIterator import SentenceIterator


class SennaFeatureExtractor(FeatureExtractor):
    """docstring for FeatureExtractor"""
    def __init__(self, dataset=None, infile=None, vocabfile=None, binary=False, dimen=100):
        self.model_file = infile
        self.vocab_file = vocabfile
        self.binary = binary
        self.dataset = dataset
        self.dimen = dimen

    def build(self):
        if self.model_file and self.vocab_file:
            vocabs = []
            models = []
            with open(self.vocab_file, "rb") as vocablist:
                for vocab in vocablist:
                    vocabs.append(vocab.rstrip())

            with open(self.model_file, "rb") as modellist:
                for model in modellist:
                    arr_model = model.split()
                    models.append(np.array(map(float, arr_model)))
                #modelss = models[:100]
                #vocabss = vocabs[:100]
            # build our word embedding model vectorizer
            senna_dict = dict(zip(vocabs, models))
            sentences = SentenceIterator(self.dataset)

            self.vectorizer = WordEmbeddingVectorizer(senna_dict, self.dimen)
            self.vectorizer.fit(sentences)
        else:
            pass

        return self

    def extract_existing_features(self, dataset):
        return super(SennaFeatureExtractor, self).extract_features(dataset)

    def get_name(self):
        return "SENNA C&W SSWE"
