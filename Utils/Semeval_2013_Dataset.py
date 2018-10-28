import numpy as np
import sys, os
import argparse
from Utils.twokenize import *
from Utils.WordVecs import *
from Utils.Representations import *


def conv_tweet(tweet, word_vecs):
    """
    Returns a concatenation of max, min and average
    vectors for the words in a tweet, following
    Tang et al. Learning Sentiment-Specific Word Embedding
    for Twitter Sentiment Classification
    """

    rep = []
    for w in words(tweet, word_vecs):
        try:
            rep.append(word_vecs[w])
        except KeyError:
            rep.append(word_vecs['the'])

    rep = np.array(rep)
    maxv = rep.max(axis=0)
    minv = rep.min(axis=0)
    avev = rep.mean(axis=0)

    return np.concatenate((maxv, minv, avev))

def words(sentence, model):
    """
    Returns the tokenized sentence after
    having removed mentions and urls.
    """
    return rem_mentions_urls(tokenize(sentence.lower()))

def rem_mentions_urls(tokens):
    """
    Replaces any mentions with 'at' and any url with 'url'.
    """
    final = []
    for t in tokens:
        if t.startswith('@'):
            final.append('at')
        elif t.startswith('http'):
            final.append('url')
        else:
            final.append(t)
    return final

class Semeval_Dataset():
    """
    This class converts the annotated SemEval 2013 task 2 data, which
    was previously downloaded, into an abstract dataset class, which
    the classifiers use.

    DIR:    The directory where the files train.tsv, dev.tsv, and test.tsv are found.
    model:  The WordVec model, which holds the word embeddings to be tested.
    one_hot:If True, the y labels will be a one-hot vector (for use with Keras models),
            otherwise, it will just be the class label.
    binary: If True, only positive and negative examples will be used,
            otherwise, positive, neutral, and negative examples are included.
    rep:    The representation of an example, i.e. ave_vecs, sum_vecs, conv_tweet, words, etc.

    """

    def __init__(self, DIR, model, one_hot=True, 
                 binary=False, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot
        self.binary = binary

        Xtrain, Xdev, Xtest, ytrain, ydev,  ytest = self.open_data(DIR, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest

    def to_array(self, y, N):
        '''
        converts an integer-based class into a one-hot array
        y = the class integer
        N = the number of classes
        '''
        return np.eye(N)[y]

    def convert_ys(self, y):
        if 'negative' in y:
            return 0
        elif 'neutral' in y:
            return 1
        elif 'objective' in y:
            return 1
        elif 'positive' in y:
            if self.binary:
                return 1
            else:
                return 2

    def open_data(self, DIR, model, rep):
        """
        Opens the data from DIR and changes the instances
        to the representation determined by rep and model.
        Finally, these are split into train, dev, and test
        splits.
        """

        train = []
        for line in open(os.path.join(DIR, 'train.tsv'), encoding="utf8"):
            idx, sidx, label, tweet = line.split('\t')
            if self.binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    train.append((label, tweet))
            else:
                train.append((label, tweet))

        dev = []
        for line in open(os.path.join(DIR, 'dev.tsv'), encoding="utf8"):
            idx, sidx, label, tweet = line.split('\t')
            if self.binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    dev.append((label, tweet))
            else:
                dev.append((label, tweet))

        test = []
        for line in open(os.path.join(DIR, 'test.tsv'), encoding="utf8"):
            idx, sidx, label, tweet = line.split('\t')
            if self.binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:   
                    test.append((label, tweet))
            else:
                test.append((label, tweet))


        ytrain, Xtrain = zip(*train)
        ydev,   Xdev   = zip(*dev)
        ytest,  Xtest  = zip(*test)
                    
        Xtrain = [rep(sent, model) for sent in Xtrain]
        ytrain = [self.convert_ys(y) for y in ytrain]

        Xdev = [rep(sent, model) for sent in Xdev]
        ydev = [self.convert_ys(y) for y in ydev]

        Xtest  = [rep(sent, model) for sent in Xtest]
        ytest = [self.convert_ys(y) for y in ytest]

        if self.one_hot:
            if self.binary:
                ytrain = [self.to_array(y, 2) for y in ytrain]
                ydev = [self.to_array(y,2) for y in ydev]
                ytest = [self.to_array(y,2) for y in ytest]
            else:
                ytrain = [self.to_array(y, 3) for y in ytrain]
                ydev = [self.to_array(y,3) for y in ydev]
                ytest = [self.to_array(y,3) for y in ytest]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)
        
        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest
