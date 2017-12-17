import csv
import re


def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    patt = re.compile(r"(.)\1{1,}", re.DOTALL)
    return patt.sub(r"\1\1", s)


def processTweet(tweet):
    # process the tweets
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'at_user', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')
    return tweet


def getStopWordList(fname):
    # read the stopwords
    stopWords = []
    stopWords.append('rt')
    stopWords.append('url')
    stopWords.append('at_user')

    fp = open(fname, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


def getFeatureVector(tweet, stopWords):
    featureVector = []
    words = tweet.split()
    for w in words:
        # replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        # ignore if it is a stopWord
        if (w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector


def tweets_prepocess(inputfile, outputfile, stopwordfile):
    with open(inputfile, 'r') as f:
        readerCSV = csv.reader(f, delimiter=',', dialect='excel')
        #print(readerCSV)
        data = []
        stopwords = getStopWordList(stopwordfile)
        for row in readerCSV:
            if len(row)>1:
                tweet = row[0]
                label = row[1]
                #print(tweet)
                tweet_processed = processTweet(replaceTwoOrMore(tweet))
                tweet_features = getFeatureVector(tweet_processed, stopwords)
                #print(tweet)
                #print("tweet")
                #print(tweet_features)
                if len(tweet_features) > 3:
                    tweet_tokens = ' '.join(tweet_features)
                    #data.append(['" {} "'.format(tweet_tokens),'" {} "'.format(label)])
                    data.append(['" {} "'.format(tweet_tokens),label])
                    #data.append([tweet_tokens, label])
    f.close()

    # preprocess and store in a new clean csv file
    with open(outputfile, "w") as fw:
        writer = csv.writer(fw, delimiter='\t', lineterminator='\n', dialect='excel', quoting=csv.QUOTE_NONE,
                            quotechar=None)
        #writer = csv.writer(fw, delimiter='\t', lineterminator='\n', dialect='excel', quoting=csv.QUOTE_NONE, quotechar=None)
        #writer = csv.writer(fw, delimiter='\t', dialect='excel-tab',quoting=csv.QUOTE_NONE, quotechar=None)
        for x in data:
            if len(x[1]) != 0:
                writer.writerow(x)
    fw.close()
