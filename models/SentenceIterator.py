class SentenceIterator(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for sentence in self.dataset:
            yield sentence.split() 

