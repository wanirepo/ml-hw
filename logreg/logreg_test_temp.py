import random
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

args = argparser.parse_args()


# FIGURE OUT PART
# PART1: train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)
open(args.vocab, 'r')
df = [float(x.split("\t")[1]) for x in open(args.vocab, 'r') if '\t' in x]
vocab = [x.split("\t")[0] for x in open(args.vocab, 'r') if '\t' in x]

train = []
test = []

class Example:
    def __init__(self, label, words, vocab, df):
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1

for label, input in [(1, args.positive), (0, args.negative)]:
    for line in open(input):
        ex = Example(label, line.split(), args.vocab, df)
        if random.random() <= test_proportion:
            test.append(ex)
        else:
            train.append(ex)

random.shuffle(train)
random.shuffle(test)

print("Read in %i train and %i test" % (len(train), len(test)))

# PART2: lr = LogReg(len(vocab), args.mu, lambda x: args.step)

lr = LogReg(len(vocab), args.mu, lambda x: args.step)

# PART3: lr.sg_update(ii,update_number)
update_number = 1

# what is this?
print args.passes

lr.sg_update(ii, update_number)
