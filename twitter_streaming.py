
# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

### Secret information for the authentication with the twitter API
ACCESS_TOKEN = '172760940-WFyHDrF0gwA0aj7snvx8KFuaUgyG8L6hB3Wndi9b'
ACCESS_SECRET = 'qW74TbTr6O98zEVn91tc2J8jkiSAAMRO7k91WzD4yx5MH'
CONSUMER_KEY = 'U5gIP4mJsHVIB1GKTLglHeOoE'
CONSUMER_SECRET = 'jm1lPsrRa7QYoQKW7thbN3stqWV8e6wl7ybM3AgqWJMoIMFAkq'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter_stream = TwitterStream(auth=oauth)
iterator = twitter_stream.statuses.sample(language="en")


#specify the tweets number to collect
tweet_count = 10
keys = []
for tweet in iterator:
    if 'extended_tweet' in tweet.keys():
        tweet_count -= 1
    # Twitter Python Tool wraps the data returned by Twitter
    # as a TwitterDictResponse object.
    # We convert it back to the JSON format to print/score
        print json.dumps(tweet)




    # The command below will do pretty printing for JSON data, try it out
    # print json.dumps(tweet, indent=4)

    if tweet_count <= 0:

        break

