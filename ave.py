import sys, os
import argparse
from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *
from Utils.MyMetrics import *
from Utils.WordVecs import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle
import tabulate

def print_prediction(file, prediction):
    with open(file, 'w') as out:
        for line in prediction:
            out.write(str(line) + '\n')

def get_best_C(dataset):
    """
    Find the best parameters on the dev set.
    """
    best_f1 = 0
    best_c = 0

    labels = sorted(set(dataset._ytrain))

    test_cs = [0.001, 0.003, 0.006, 0.009,
                   0.01,  0.03,  0.06,  0.09,
                   0.1,   0.3,   0.6,   0.9,
                   1,       3,    6,     9,
                   10,      30,   60,    90]
    for i, c in enumerate(test_cs):

        sys.stdout.write('\rRunning cross-validation: {0} of {1}'.format(i+1, len(test_cs)))
        sys.stdout.flush()

        clf = LogisticRegression(C=c)
        h = clf.fit(dataset._Xtrain, dataset._ytrain)
        pred = clf.predict(dataset._Xdev)
        if len(labels) == 2:
            dev_f1 = f1_score(dataset._ydev, pred, pos_label=1)
        else:
            dev_f1 = f1_score(dataset._ydev, pred, labels=labels, average='micro')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_c = c

    print()
    print('Best F1 on dev data: {0:.3f}'.format(best_f1))
    print('Best C on dev data: {0}'.format(best_c))

    return best_c, best_f1


def test_embeddings(embedding_file, file_type):
    """
	embedding_file: the word embeddings file
	file_type:		word2vec, glove, tang, bin

	Use averaged word embeddings for each word in a text as features
	for l2 regularized logistiic regression. We test the embeddings
	on 10 benchmarks.

	Stanford Sentiment corpus (Socher et al., 2013)
	OpeNER corpus (Agerri et al., 2016)
	Sentube Corpora (Severyn et al., 2016)
	Semeval 2013 twitter corpus - task 2
	

    """

    print('importing vectors...')
    vecs = WordVecs(embedding_file, file_type)
    dim = vecs.vector_size

    print('importing datasets...')
    st_fine = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            vecs,
                                            one_hot=False,
                                            binary=False,
                                            rep=ave_vecs)

    st_binary = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            vecs,
                                            one_hot=False,
                                            binary=True,
                                            rep=ave_vecs)

    opener_dataset = General_Dataset('datasets/opener',
                                     vecs,
                                     one_hot=False,
                                     rep=ave_vecs)

    sentube_auto_dataset = General_Dataset('datasets/SenTube/auto',
                                           vecs, rep=ave_vecs,
                                           binary=True,
                                           one_hot=False)

    sentube_tablets_dataset = General_Dataset('datasets/SenTube/tablets',
                                              vecs, rep=ave_vecs,
                                              binary=True,
                                              one_hot=False)

    semeval_dataset = Semeval_Dataset('datasets/semeval',
                                                vecs, rep=ave_vecs,
                                                one_hot=False)

    datasets = [st_fine, st_binary, opener_dataset, 
                sentube_auto_dataset, sentube_tablets_dataset, semeval_dataset]

    names = ['sst_fine', 'sst_binary', 'opener',
             'sentube_auto', 'sentube_tablets', 'semeval']

    # Collect results here
    results = []

    for name, dataset in zip(names, datasets):
        print('Testing vectors on {0}...'.format(name))

        # Get best parameters
        best_c, best_f1 = get_best_C(dataset)

        # Get predictions
        classifier = LogisticRegression(C=best_c)
        history = classifier.fit(dataset._Xtrain, dataset._ytrain)
        pred = classifier.predict(dataset._Xtest)
        predictions_file = "predictions/ave/" + name + '/pred.txt'
        print_prediction(predictions_file, pred)

        # Get results
        labels = sorted(set(dataset._ytrain))
        if len(labels) == 2:
            average = 'binary'
        else:
            average = 'micro'
        mm = MyMetrics(dataset._ytest, pred, labels=labels,
                       average=average, one_hot=False)
        acc, precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])

    # Add overall results
    results.append(list(np.array(results).mean(axis=0)))
    names.append('overall')

    return names, results, dim


def print_results(file, out_file, file_type):
    """
    file:       word embedding file
    out_file:   if provided, where to write results
    file_type:  word2vec, glove, tang, or bin
    """

    names, results, dim = test_embeddings(file, file_type)

    table_data = [[name] + result for name, result in zip(names, results)]
    table = tabulate.tabulate(table_data,
                              headers=['dataset', 'acc', 'prec', 'rec', 'f1'],
                              tablefmt='simple', floatfmt='.3f')

    if out_file:
        with open(out_file, 'a') as f:
            f.write('\n')
            f.write('+++Average word vectors+++\n')
            f.write(table)
            f.write('\n')
    else:
        print()
        print('+++Average word vectors+++')
        print(table)
        
def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-emb', help='location of embeddings', 
        default='embeddings/amazon-sg-100-window10-sample1e-4-negative5.txt')
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

    print('Averaged word vectors from %s + Logistic Regression' % embedding_file)

    if printout:
        print_results(embedding_file, None, file_type)
    else:
        print_results(embedding_file, output, file_type)


if __name__ == '__main__':

    args = sys.argv
    main(args)
