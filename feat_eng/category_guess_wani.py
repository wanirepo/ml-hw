from csv import DictReader, DictWriter

import re
import random
import numpy as np
from numpy import array

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

random.seed(random.random())


class LemmaTokenizer:
    def __init__(self):
	self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
	return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class Featurizer:
    def __init__(self):
        # self.vectorizer = CountVectorizer()
	#self.vectorizer = TfidfVectorizer(stop_words='english', 
        #                  ngram_range=(1,2), min_df=.001, smooth_idf=True,
        #                  use_idf=True, sublinear_tf=True
        
        #self.vectorizer = TfidfVectorizer(stop_words='english', 
        #                  ngram_range=(1,2), min_df=.001, norm='l2',
        #                  use_idf=True, smooth_idf=True, preprocessor=remove_number)
	#self.vectorizer = CountVectorizer(analyzer='word',strip_accents='ascii',
        #                    preprocessor=remove_spec,tokenizer=LemmaTokenizer(),
        #                    stop_words='english',ngram_range=(1,2))
        self.vectorizer = CountVectorizer(analyzer='word',strip_accents='ascii',
                            preprocessor=remove_spec,tokenizer=LemmaTokenizer(),
                            stop_words='english',ngram_range=(1,2))

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)
	
    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-20:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

def remove_spec(str):
    #str = str.lower()
    #str = str.replace('\'s', '')
    #str = str.replace('\n', '').replace('\t', '')
    #str = str.replace('this', '').replace('these', '')
    #str = str.replace(']', '').replace('[', '')
    #str = str.replace('point', '').replace('wa', '')
    
    #punctuation = re.compile(r'[-.?!,"*0-9]')
    punctuation = re.compile(r'[,.?!|0-9]')
    str = punctuation.sub('', str)

    #shortword = re.compile(r'\W*\b\w{1,2}\b')
    #str = shortword.sub('', str)
    return str

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])

    random.shuffle(train)
    
    l = len(train)
    
    x_train = feat.train_feature(x['text'] for x in train[:-5000])

    x_test_in = feat.test_feature(x['text'] for x in train[-5000:l])
    x_test = feat.test_feature(x['text'] for x in test)

    y_train = array(list(labels.index(x['cat']) for x in train[:-5000]))
    y_test_in = array(list(labels.index(x['cat']) for x in train[-5000:l]))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions_in = lr.predict(x_test_in)

    print("Accuracy: %f" % accuracy_score(y_test_in, predictions_in))

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
