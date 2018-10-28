import json
import sys, os
from Utils.Datasets import General_Dataset
from Utils.WordVecs import *
from Utils.Representations import *


class SenTube_Dataset(General_Dataset):

    def open_data(self, DIR, model, binary, rep):

        files = os.listdir(DIR)
        
        pos = []
        neg = []

        for file in files:
            full_filename = os.path.join(DIR, file)
            with open(full_filename) as f:
                video = json.load(f)

            for comment in video['comments']:
                if 'annotation' in comment:
                    if 'negative-product' in comment['annotation']:
                        neg.append(comment['text'])
                        
            for comment in video['comments']:
                if 'annotation' in comment:
                    if 'positive-product' in comment['annotation']:
                        pos.append(comment['text'])

        posy = [1] * len(pos)
        negy = [0] * len(neg)
        
        pos = list(zip(posy, pos))
        neg = list(zip(negy, neg))

        pos_train_idx = int(len(pos) * .75)
        pos_dev_idx = int(len(pos) * .8)

        neg_train_idx = int(len(neg) * .75)
        neg_dev_idx = int(len(neg) * .8)

        train_neg = neg[:neg_train_idx]
        dev_neg = neg[neg_train_idx:neg_dev_idx]
        test_neg = neg[neg_dev_idx:]

        train_pos = pos[:pos_train_idx]
        dev_pos = pos[pos_train_idx:pos_dev_idx]
        test_pos = pos[pos_dev_idx:]

        train_data = train_pos + train_neg
        dev_data = dev_pos + dev_neg
        test_data = test_pos + test_neg

        ytrain, Xtrain = zip(*train_data)
        Xtrain = [rep(sent, model) for sent in Xtrain]
        
        ydev, Xdev = zip(*dev_data)
        Xdev = [rep(sent, model) for sent in Xdev]

        ytest, Xtest = zip(*test_data)
        Xtest = [rep(sent, model) for sent in Xtest]

        if self.one_hot:
            ytrain = [self.to_array(i,2) for i in ytrain]
            ydev =   [self.to_array(i,2) for i in ydev]
            ytest =  [self.to_array(i,2) for i in ytest]

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
            
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest
            
            
