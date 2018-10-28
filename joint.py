import numpy as np
import sys, os
import argparse
import tabulate
from Utils.twokenize import *
from Utils.MyMetrics import *
from Utils.WordVecs import *
from Utils.Representations import *
from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def print_prediction(file, prediction):
    with open(file, 'w') as out:
        for line in prediction:
            out.write(str(line) + '\n')


def get_best_C(Xtrain, ytrain, Xdev, ydev):
    """
    Find the best parameters on the dev set.
    """
    best_f1 = 0
    best_c = 0

    labels = sorted(set(ytrain))

    test_cs = [0.001, 0.0025, 0.005, 0.0075,
              0.01, 0.025, 0.05, 0.075,
              0.1, 0.25, 0.5, 0.75,
              1, 2.5, 5, 7.5]
    for i, c in enumerate(test_cs):

        sys.stdout.write('\rRunning cross-validation: {0} of {1}'.format(i+1, len(test_cs)))
        sys.stdout.flush()

        clf = LogisticRegression(C=c)
        h = clf.fit(Xtrain, ytrain)
        pred = clf.predict(Xdev)
        if len(labels) == 2:
            dev_f1 = f1_score(ydev, pred, pos_label=1)
        else:
            dev_f1 = f1_score(ydev, pred, labels=labels, average='micro')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_c = c

    print()
    print('Best F1 on dev data: {0:.3f}'.format(best_f1))
    print('Best C on dev data: {0}'.format(best_c))
    
    return best_c, best_f1

def test_embeddings(embedding_file, file_type):
    """
    Tang et al. (2014) embeddings and cassification approach
    on a number of benchmark datasets.

    """

    print('importing vectors...')
    vecs = WordVecs(embedding_file, file_type)
    dim = vecs.vector_size

    print('Importing datasets...')
    st_fine = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            None,
                                            one_hot=False,
                                            binary=False,
                                            rep=words)
    

    st_binary = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            None,
                                            one_hot=False,
                                            binary=True,
                                            rep=words)

    opener_dataset = General_Dataset('datasets/opener',
                                     vecs,
                                     one_hot=False,
                                     rep=words)

    sentube_auto_dataset = General_Dataset('datasets/SenTube/auto',
                                           vecs._w2idx, rep=words,
                                           binary=True,
                                           one_hot=False)

    sentube_tablets_dataset = General_Dataset('datasets/SenTube/tablets',
                                              vecs._w2idx, rep=words,
                                              binary=True,
                                              one_hot=False)

    semeval_dataset = Semeval_Dataset('datasets/semeval',
                                                vecs._w2idx, rep=words,
                                                one_hot=False)

    datasets = [st_fine, st_binary, opener_dataset, 
                sentube_auto_dataset, sentube_tablets_dataset, semeval_dataset]


    names = ['sst_fine', 'sst_binary', 'opener',
             'sentube_auto', 'sentube_tablets', 'semeval']

    # Collect results here
    results = []

    for name, dataset in zip(names, datasets):
        print('Testing on {0}...'.format(name))

        Xtrain = np.array([conv_tweet(' '.join(t), vecs) for t in dataset._Xtrain])
        Xtest = np.array([conv_tweet(' '.join(t), vecs) for t in dataset._Xtest])
        Xdev = np.array([conv_tweet(' '.join(t), vecs) for t in dataset._Xdev])

        # get best parameters on dev set
        best_C, best_rate = get_best_C(Xtrain, dataset._ytrain,
                                       Xdev, dataset._ydev)


        clf = LogisticRegression(C=best_C)
        h = clf.fit(Xtrain, dataset._ytrain)
        pred = clf.predict(Xtest)
        predictions_file = "predictions/joint/" + name + '/pred.txt'
        print_prediction(predictions_file, pred)

        labels = sorted(set(dataset._ytrain))
        if len(labels) == 2:
            average = 'binary'
        else:
            average = 'micro'
        mm = MyMetrics(dataset._ytest, pred, one_hot=False, labels=labels, average=average)
        acc, precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])

    results.append(list(np.array(results).mean(axis=0)))
    names.append('overall')

    return names, results, dim


def print_results(file, out_file, file_type):

    names, results, dim = test_embeddings(file, file_type)

    table_data = [[name] + result for name, result in zip(names, results)]
    table = tabulate.tabulate(table_data, headers=['dataset', 'acc', 'prec', 'rec', 'f1'], tablefmt='simple', floatfmt='.3f')

    if out_file:
        with open(out_file, 'a') as f:
            f.write('\n')
            f.write('+++Joint+++\n')
            f.write(table)
            f.write('\n')
    else:
        print()
        print('+++Joint+++')
        print(table)
        
def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-emb', help='location of embeddings', 
        default='embeddings/sswe-u-50.txt')
    parser.add_argument('-file_type', help='glove style embeddings or word2vec style: default is w2v',
        default='word2vec')
    parser.add_argument('-output', help='output file for results', default='./results.txt')
    parser.add_argument('-printout', help='instead of printing to file, print to sysout',
                        type=bool, default=False)

    args = vars(parser.parse_args())
    embedding_file = args['emb']
    file_type = args['file_type']
    output = args['output']
    printout = args['printout']

    print('testing on %s' % embedding_file)

    if printout:
        print_results(embedding_file, None, file_type)
    else:
        print_results(embedding_file, output, file_type)

if __name__ == '__main__':

    args = sys.argv
    main(args)

