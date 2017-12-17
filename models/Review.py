class Review(object):
    num_review = 0
    uid_placeholder = 1 
    """docstring for Review"""

    def __init__(self, content, polarity):
        super(Review, self).__init__()
        self.content = content
        self.polarity = polarity
        Review.num_review += 1
        self.id = Review.num_review

    def to_string(self):
        return str(self.id) + "\t" + str(Review.uid_placeholder) + "\t" + self.polarity + "\t" + self.content
