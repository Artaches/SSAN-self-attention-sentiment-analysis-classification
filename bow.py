import sys
import argparse
import tabulate
from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.MyMetrics import *
from Utils.SenTube_Dataset import *
from Utils.Semeval_2013_Dataset import *
from sklearn.linear_model import LogisticRegression


def print_prediction(file, prediction):
    # Print the predictions to a file for later reference.
    with open(file, 'w') as out:
        for line in prediction:
            out.write(str(line) + '\n')

def bow(sent, vocab):
    # Create bag of word representations for the sentence.
    s = np.zeros((len(vocab)))
    for w in sent:
        idx = vocab[w]
        s[idx] += 1
    return s


def test():
    """
    Runs the test from Barnes et al, 2017 
    """

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
                                     None,
                                     one_hot=False,
                                     rep=words)


    sentube_auto_dataset = General_Dataset('datasets/SenTube/auto',
                                           None, rep=words,
                                           binary=True,
                                           one_hot=False)

    sentube_tablets_dataset = General_Dataset('datasets/SenTube/tablets',
                                              None, rep=words,
                                              binary=True,
                                              one_hot=False)

    semeval_dataset = Semeval_Dataset('datasets/semeval',
                                                None, rep=words,
                                                one_hot=False)

    datasets = [st_fine, st_binary, opener_dataset, 
                sentube_auto_dataset, sentube_tablets_dataset, semeval_dataset]

    names = ['sst_fine', 'sst_binary', 'opener',
             'sentube_auto', 'sentube_tablets', 'semeval']

    # Collect results here
    results = []


    for name, dataset in zip(names, datasets):
        print('Testing on {0}...'.format(name))
        dataset._Xtrain = list(dataset._Xtrain)
        dataset._Xtest = list(dataset._Xtest)
        dataset._Xdev = list(dataset._Xdev)

        vocab = {}
        for sent in dataset._Xtrain + dataset._Xdev + dataset._Xtest:
            for w in sent:
                if w not in vocab:
                    vocab[w] = len(vocab)

        trainX = [bow(s, vocab) for s in dataset._Xtrain]
        trainy = dataset._ytrain

        devX = [bow(s, vocab) for s in dataset._Xdev]
        devy = dataset._ydev

        testX = [bow(s, vocab) for s in dataset._Xtest]
        testy = dataset._ytest

        clf = LogisticRegression(C=1)
        h = clf.fit(trainX, trainy)
        pred = clf.predict(testX)

        predictions_file = "predictions/bow/" + name + '/pred.txt'
        print_prediction(predictions_file, pred)
        
        labels = sorted(set(dataset._ytrain))
        if len(labels) == 2:
            average = 'binary'
        else:
            average = 'macro'
        mm = MyMetrics(dataset._ytest, pred, labels=labels, average=average, one_hot=False)
        acc, precision, recall, f1 = mm.get_scores()
        results.append([acc, precision, recall, f1])

    results.append(list(np.array(results).mean(axis=0)))
    names.append('overall')
    
    return names, results


def print_results(out_file):

    names, results = test()

    table_data = [[name] + result for name, result in zip(names, results)]
    table = tabulate.tabulate(table_data, headers=['dataset', 'acc', 'prec', 'rec', 'f1'], tablefmt='simple', floatfmt='.3f')

    if out_file:
        with open(out_file, 'a') as f:
            f.write('\n')
            f.write('+++Bag-of-words+++\n')
            f.write(table)
            f.write('\n')
    else:
        print()
        print('+++Bag-of-words with L2 regularized Logistic Regression+++')
        print(table)
        
def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-output', help='output file for results', default='./results.txt')
    parser.add_argument('-printout', help='instead of printing to file, print to sysout',
                        type=bool, default=False)

    args = vars(parser.parse_args())
    output = args['output']
    printout = args['printout']

    print('Bag of words + Logistic Regression...')

    if printout == True:
        print_results(None)
    else:
        print_results(output)

    print()


if __name__ == '__main__':

    args = sys.argv
    main(args)

