from pymongo import MongoClient
import json
try:
    import json
except ImportError:
    import simplejson as json
import logging
logger = logging.Logger('catch_all')

client = MongoClient('localhost', 27017)
db = client.twitter_db

tweet_filename  = "/root/PycharmProjects/untitled1/full_tweet.txt"
tweet_file = open(tweet_filename,'r')

for line in tweet_file:
    try:
        tweet = json.loads(line.strip())
        print(type(json))
        #posts = db.posts
        #posts.insert(post)
    except Exception as e:
        # read in a line is not in JSON format (sometimes error occured)
       logger.error(e, exc_info=True)
       print(e)