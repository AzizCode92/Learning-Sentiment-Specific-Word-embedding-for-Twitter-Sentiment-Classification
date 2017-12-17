# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json
import logging
from pymongo import MongoClient

logger = logging.Logger('catch_all')

tweet_filename  = "/root/PycharmProjects/project_text_mining/tweets.txt"
tweet_file = open(tweet_filename,'r')
keys_to_delete = ["favorited","contributors","truncated","possibly_sensitive","is_quote_status"\
                  ,"in_reply_to_status_id","filter_level","geo","favorite_count","extended_tweet"\
                  ,"entities","in_reply_to_user_id_str","retweeted","coordinates","timestamp_ms"\
                  ,"source","in_reply_to_status_id_str","in_reply_to_screen_name","display_text_range"\
                  ,"place","retweet_count","in_reply_to_user_id","user","_id"]

client = MongoClient('localhost', 27017)
db = client.twitter_db


for line in tweet_file:
    try:
        tweet = json.loads(line.strip())
        tweet["text"] = tweet["extended_tweet"]["full_text"].encode("utf-8")
        for key in keys_to_delete:
            if key in tweet:
                del tweet[key]

        posts = db.twitter_db
        posts.insert(tweet)
        print(tweet)

    except Exception as e:
        # read in a line is not in JSON format (sometimes error occured)
        logger.error(e, exc_info=True)
        print(e)
tweet_file.close()



"""

try :
    tweet = json.loads(line.strip())
    keys.append(tweet.keys())
    new_dict = {"text": tweet["text"] for key in keys}
    print(new_dict)
    #dict_i_want = {key: tweet[key] for key in keys}
    #print dict_i_want
    #if 'extended_tweet' in tweet.keys()  :
    #print tweet['extended_tweet']['full_text'].encode('utf-8')  # content of the tweet
except Exception as e:
    # read in a line is not in JSON format (sometimes error occured)
    logger.error(e, exc_info=True)
    print(e)
tweet_file.close()
"""