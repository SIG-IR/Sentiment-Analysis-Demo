import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize, tokenize
from nltk.corpus import stopwords

# Using BeautifulSoup to extract raw text
html = requests.get('https://thearcmag.com/why-im-voting-for-trump-d86b1786fbbe')
soup = BeautifulSoup(html.text, "html.parser")
raw_text = soup.findAll("div", { "class": "layoutSingleColumn" })[1].getText()

# Getting stop words
stop_words = list(stopwords.words('english'))
print("Stop words {}\n\n\n\n".format(stop_words))

# Tokenizing the words
tokens = word_tokenize(raw_text)
print("{} tokens identified".format(len(tokens)))

# Removing stop words from tokenized list
tokens_without_stop_words = [t for t in tokens if t not in stop_words]
print("{} real tokens: \n\n\n\n\n{}".format(len(tokens_without_stop_words), tokens_without_stop_words))

"""
WIP

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import names
from nltk.sentiment.util import *
    
# Using already trained
sid = SentimentIntensityAnalyzer()
sentences = tokenize.sent_tokenize(raw_text)
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print '{0}: {1}, '.format(k, ss[k]),
    print("\n")


labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))
print(nltk.classify.accuracy(classifier, test_set))
"""