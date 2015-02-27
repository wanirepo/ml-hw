from csv import DictReader, DictWriter

import re
import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer:
    def __init__(self):
	self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
	return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class Featurizer:
    def __init__(self):
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
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

def remove_spec(str):
    punctuation = re.compile(r'[-.?!|0-9]')
    str = punctuation.sub('', str)

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

    x_train = feat.train_feature(x['text'] for x in train)
    x_test = feat.test_feature(x['text'] for x in test)

    y_train = array(list(labels.index(x['cat']) for x in train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
