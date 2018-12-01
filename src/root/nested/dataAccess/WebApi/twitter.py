import tweepy
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

class twitter(object):
    """description of class"""

    def __init__(self, **kwargs):

        self.access_token = "2295104966-YyMcXmJxWKRM7ftVrVzlc6Xa50RfKmTBIoIxVjj"
        self.access_token_secret = "FAtvOEHCZti7MkcXTblLZt1IOMRRyt4OmoqNR5xOAvZLj"
        self.consumer_key = "zF2itLSqmns4yBfSsL52FSM9Z"
        self.consumer_secret = "Bl8wFVLOTaH46Oc52BH70L66UPHuGc8iVi9FtEytu8Q688F6sW"

        return super().__init__(**kwargs)

    def init_tweepy_auth_handler(self):

        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)

        return (auth)

    def create_twitter_stream_obj(self,
                                  auth,
                                  tweet_output_file,
                                  filter_for_list = None):

        stream_listener = MyStreamListener(tweet_output_file = tweet_output_file)
        stream = tweepy.Stream(auth, stream_listener)
        stream.filter(track = filter_for_list)
        return (stream)

    def read_tweets(self,
                    tweets_output_file):

        list_of_tweets = []
        tweets_file = open(tweets_output_file, "r")
        for line in tweets_file:
            tweet = json.loads(line)
            list_of_tweets.append(tweet)
        tweets_file.close()

        return (list_of_tweets)

    def tweet_list_to_df(self,
                         list_of_tweets,
                         list_col_names):

        # example of list_col_names from Datacamp.com example is ['text', 'lang']
        # the text of the tweet, the language tweet written in.
        df = pd.DataFrame(list_of_tweets, columns=list_col_names)
        return (df)

    def word_in_text(self,
                     word,
                     text):

        # use this function to look for words in text of a tweet.

        word = word.lower()
        text = text.lower()
        match = re.search(word, text)
        if match:
            return True
        return False

class MyStreamListener(tweepy.StreamListener):

    def __init__(self, tweet_output_file, api=None):

        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open(tweet_output_file, "w")

    def on_status(self, status):

        tweet = status._json
        self.file.write(json.dumps(tweet) + "\n")
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()


twitter_obj = twitter()
auth = twitter_obj.init_tweepy_auth_handler()
twitter_stream = twitter_obj.create_twitter_stream_obj(auth, 
                                      "C:\\Users\\ghazy\\workspace\\data\\twitter_api\\tweets_ex.csv",
                                      ['clinton', 'trump', 'sanders', 'cruz'])
list_of_tweets = twitter_obj.read_tweets("C:\\Users\\ghazy\\workspace\\data\\twitter_api\\tweets_ex.csv")
df = twitter_obj.tweet_list_to_df(list_of_tweets,
                             ['text', 'lang'])

[clinton, trump, sanders, cruz] = [0,0,0,0]
for index, row in df.iterrows():
    clinton += twitter_obj.word_in_text('clinton', row['text'])
    trump += twitter_obj.word_in_text('trump', row['text'])
    sanders += twitter_obj.word_in_text('sanders', row['text'])
    cruz += twitter_obj.word_in_text('cruz', row['text'])

import seaborn as sns
sns.set(color_codes = True)
xlabels = ['clinton', 'trump', 'sanders', 'cruz']
ax = sns.barplot(xlabels, [clinton, trump, sanders, cruz])
ax.set(ylabel = "count")
plt.show()