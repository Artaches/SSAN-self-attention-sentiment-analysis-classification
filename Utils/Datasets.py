import os, re
import numpy as np
from Utils.Representations import *
#from spell_checker import *

class General_Dataset(object):
    """This class takes as input the directory of a corpus annotated for 4 levels
    sentiment. This directory should have 4 .txt files: strneg.txt, neg.txt,
    pos.txt and strpos.txt. It also requires a word embedding model, such as
    those used in word2vec or GloVe.

    binary: instead of 4 classes you have binary (pos/neg). Default is False

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each vector. Default is np.float32.

    rep: this determines how the word vectors are represented.

         sum_vecs: each sentence is represented by one vector, which is
                    the sum of each of the word vectors in the sentence.

         ave_vecs: each sentence is represented as the average of all of the
                    word vectors in the sentence.

         idx_vecs: each sentence is respresented as a list of word ids given by
                    the word-2-idx dictionary.
    """

    def __init__(self, DIR, model, binary=False, one_hot=True,
                 dtype=np.float32, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, binary, rep)


        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def open_data(self, DIR, model, binary, rep):
        if binary:
            ##################
            # Binary         #
            ##################
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg
            devdata = dev_pos + dev_neg
            testdata = test_pos + test_neg
            # Training data
            Xtrain = [data for data, y in traindata]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 2) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.one_hot is True:
                ydev = [self.to_array(y, 2) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]

            # Test data
            Xtest = [data for data, y in testdata]
            if self.one_hot is True:
                ytest = [self.to_array(y, 2) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
        
        else:
            ##################
            # 4 CLASS        #
            ##################
            train_strneg = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_strpos = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  3, model, encoding='latin',
                                  representation=rep)
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  2, model, encoding='latin',
                                  representation=rep)
            dev_strneg = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_strpos = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                3, model, encoding='latin',
                                representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                2, model, encoding='latin',
                                representation=rep)
            test_strneg = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_strpos = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 3, model, encoding='latin',
                                 representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 2, model, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg + train_strneg + train_strpos
            devdata = dev_pos + dev_neg + dev_strneg + dev_strpos
            testdata = test_pos + test_neg + test_strneg + test_strpos


            # Training data
            Xtrain = [data for data, y in traindata]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 4) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.one_hot is True:
                ydev = [self.to_array(y, 4) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]

            # Test data
            Xtest = [data for data, y in testdata]
            if self.one_hot is True:
                ytest = [self.to_array(y, 4) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

class Stanford_Sentiment_Dataset(object):
    """Stanford Sentiment Treebank
    """

    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, binary=False, rep=ave_vecs):

        self.rep = rep
        self.one_hot = one_hot
        self.binary = binary

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, rep)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def flatten(self, tree_sent):
        """
        Flattens constituency trees to get just the tokens.
        """
        label = int(tree_sent[1])
        text = re.sub('\([0-9]', ' ', tree_sent).replace(')','').split()
        return label, ' '.join(text)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def remove_neutral(self, data):
        """
        Removes any neutral examples
        from the data for binary
        classification.
        """
        final = []
        for y, x in data:
            if y in [0, 1]:
                final.append((0, x))
            elif y in [3, 4]:
                final.append((1, x))
        return final

    def open_data(self, DIR, model, rep):
        """
        Takes the directory DIR where the files
        train.txt, dev.txt, and test.txt are found.
        It returns the final representations of
        the instances, given the word vectors or
        word-to-index map in
        'model' and the representation in 'rep'.
        """
        train = open(os.path.join(DIR, 'train.txt'))
        dev = open(os.path.join(DIR, 'dev.txt'))
        test = open(os.path.join(DIR, 'test.txt'))

        train_data = [self.flatten(x) for x in train]
        if self.binary:
            train_data = self.remove_neutral(train_data)
        ytrain, Xtrain = zip(*train_data)
        Xtrain = [rep(sent, model) for sent in Xtrain]

        dev_data = [self.flatten(x) for x in dev]
        if self.binary:
            dev_data = self.remove_neutral(dev_data)
        ydev, Xdev = zip(*dev_data)
        Xdev = [rep(sent, model) for sent in Xdev]

        test_data = [self.flatten(x) for x in test]
        if self.binary:
            test_data = self.remove_neutral(test_data)
        ytest, Xtest = zip(*test_data)
        Xtest = [rep(sent, model) for sent in Xtest]

        if self.one_hot is True:
            if self.binary:
                ytrain = [self.to_array(y, 2) for y in ytrain]
                ydev = [self.to_array(y, 2) for y in ydev]
                ytest = [self.to_array(y, 2) for y in ytest]
            else:
                ytrain = [self.to_array(y, 5) for y in ytrain]
                ydev = [self.to_array(y, 5) for y in ydev]
                ytest = [self.to_array(y, 5) for y in ytest]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest
