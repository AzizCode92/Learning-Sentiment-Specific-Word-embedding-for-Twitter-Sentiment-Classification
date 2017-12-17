import gensim
from FeatureExtractor import FeatureExtractor
from vectorizer.WordEmbeddingVectorizer import WordEmbeddingVectorizer

from models.SentenceIterator import SentenceIterator


class WordEmbeddingFeatureExtractor(FeatureExtractor):
    """docstring for FeatureExtractor"""
    def __init__(self, dataset=None, infile=None, binary=False, dimen=100, sswe=0):
        super(WordEmbeddingFeatureExtractor, self).__init__(dataset)
        self.model_file = infile
        self.binary = binary
        self.dimen = dimen
        self.sswe = sswe

    def build(self):
        if not self.model_file:
            sentences = SentenceIterator(self.dataset)
            w2v = gensim.models.Word2Vec(sentences, size=self.dimen, min_count=1)
            word_vectors = w2v.wv
            del w2v   # free memory
        else:
            word_vectors = gensim.models.KeyedVectors.load_word2vec_format(self.model_file, binary=self.binary)

            # build our word embedding model vectorizer
            # w2v_dict = dict(zip(w2v.index2word, w2v.syn0))
            sentences = SentenceIterator(self.dataset)

        self.vectorizer = WordEmbeddingVectorizer(word_vectors, self.dimen)
        self.vectorizer.fit(sentences)

        return self

    def extract_existing_features(self, dataset):
        return super(WordEmbeddingFeatureExtractor, self).extract_features(dataset)

    def save_model_to_file(self, outfile, vocabfile=None, binary=True):
        sentences = SentenceIterator(self.dataset)
        w2v = gensim.models.Word2Vec(sentences, size=self.dimen, min_count=1, sg=1, workers=4, iter=10)

        w2v.wv.save_word2vec_format(outfile, fvocab=vocabfile, binary=binary)

    def get_name(self):
        if self.sswe == 1:
            return "SSWE + Word2Vec"
        else:
            return "Gensim Word2Vec"
