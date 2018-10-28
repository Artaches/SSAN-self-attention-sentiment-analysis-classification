from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import sys
import argparse
import tabulate
import json
import re
from Utils.WordVecs import *
from Utils.MyMetrics import *
from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *

def print_prediction(file, prediction):
    with open(file, 'w') as out:
        for line in prediction:
            out.write(str(line) + '\n')

def get_dev_params(dataset_name, outfile, bi,
                   Xtrain, ytrain, Xdev, ydev, vecs):

    
    # If you have already run the dev experiment, just get results
    if os.path.isfile(outfile):
        with open(outfile) as out:
            dev_results = json.load(out)
            if dataset_name in dev_results:
                f1 = dev_results[dataset_name]['f1']
                dim = dev_results[dataset_name]['dim']
                dropout = dev_results[dataset_name]['dropout']
                epoch = dev_results[dataset_name]['epoch']
                return dim, dropout, epoch, f1



    # Otherwise, run a test on the dev set to get the best parameters
    best_f1 = 0
    best_dim = 0
    best_dropout = 0
    best_epoch = 0

    output_dim = ytrain.shape[1]
    labels = sorted(set(ytrain.argmax(1)))

    dims = np.arange(50, 300, 25)
    dropouts = np.arange(0.1, 0.6, 0.1)
    epochs = np.arange(3, 25)

    # Do a random search over the parameters
    for i in range(10):

        dim = int(dims[np.random.randint(0, len(dims))])
        dropout = float(dropouts[np.random.randint(0, len(dropouts))])
        epoch = int(epochs[np.random.randint(0, len(epochs))])

        if bi:
            clf = create_BiLSTM(vecs, dim, output_dim, dropout)
            h = clf.fit(Xtrain, ytrain, epochs=epoch, verbose=1)
        else:
            clf = create_LSTM(vecs, dim, output_dim, dropout)
            h = clf.fit(Xtrain, ytrain, epochs=epoch, verbose=1)

        pred = clf.predict(Xdev, verbose=0)
        if len(labels) == 2:
            mm = MyMetrics(ydev, pred, one_hot=True, labels=labels, average='binary')
            _, _, _, dev_f1 = mm.get_scores()
        else:
            mm = MyMetrics(ydev, pred, one_hot=True, labels=labels, average='micro')
            _, _, _, dev_f1 = mm.get_scores()
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_dim = dim
            best_dropout = dropout
            best_epoch = epoch
        print('new best f1: {0:.3f} dim:{1} dropout:{2} epochs:{3}'.format(best_f1, dim, dropout, epoch))

        if os.path.isfile(outfile):
            with open(outfile) as out:
                dev_results = json.load(out)
                dev_results[dataset_name] = {'f1': best_f1,
                         'dim': best_dim,
                         'dropout': best_dropout,
                         'epoch': best_epoch}
            with open(outfile, 'w') as out:
                json.dump(dev_results, out)

        else:
            dev_results = {}
            dev_results[dataset_name] = {'f1': best_f1,
                         'dim': best_dim,
                         'dropout': best_dropout,
                         'epoch': best_epoch}
            with open(outfile, 'w') as out:
                json.dump(dev_results, out)

    return best_dim, best_dropout, best_epoch, best_f1
    

def add_unknown_words(wordvecs, vocab, min_df=1, dim=50):
    """
    For words that occur at least min_df, create a separate word vector
    0.25 is chosen so the unk vectors have approximately the same variance
    as pretrained ones
    """
    for word in vocab:
        if word not in wordvecs and vocab[word] >= min_df:
            wordvecs[word] = np.random.uniform(-0.25, 0.25, dim)


def get_W(wordvecs, dim=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(wordvecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, dim), dtype='float32')
    W[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in wordvecs:
        W[i] = wordvecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def create_LSTM(wordvecs, lstm_dim=300, output_dim=2, dropout=.5,
                weights=None, train=True):
    """
    Create simple one layer lstm
    lstm_dim: dimension of hidden layer
    dropout: 0-1
    weights: if you have pretrained embeddings, you can include them here
    train: if true, updates the original word embeddings
    """

    model = Sequential()
    if weights != None:
        model.add(Embedding(len(wordvecs)+1,
            len(wordvecs['the']),
            weights=[weights],
                    trainable=train))
    else:
        model.add(Embedding(len(wordvecs)+1,
            len(wordvecs['the']),
                    trainable=train))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_dim))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_BiLSTM(wordvecs, lstm_dim=300, output_dim=2, dropout=.5,
                weights=None, train=True):
    model = Sequential()
    if weights != None:
        model.add(Embedding(len(wordvecs)+1,
            len(wordvecs['the']),
            weights=[weights],
                    trainable=train))
    else:
        model.add(Embedding(len(wordvecs)+1,
            len(wordvecs['the']),
                    trainable=train))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(lstm_dim)))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def idx_sent(sent, w2idx):
    return np.array([w2idx[w] for w in sent])

def convert_dataset(dataset, w2idx, maxlen=50):
    dataset._Xtrain = np.array([idx_sent(s, w2idx) for s in dataset._Xtrain])
    dataset._Xdev = np.array([idx_sent(s, w2idx) for s in dataset._Xdev])
    dataset._Xtest = np.array([idx_sent(s, w2idx) for s in dataset._Xtest])
    dataset._Xtrain = pad_sequences(dataset._Xtrain, maxlen)
    dataset._Xdev = pad_sequences(dataset._Xdev, maxlen)
    dataset._Xtest = pad_sequences(dataset._Xtest, maxlen)
    return dataset

def write_vecs(matrix, w2idx, outfile):
    vocab = sorted(w2idx.keys())
    with open(outfile, 'w') as out:
        for w in vocab:
            try:
                out.write(w + ' ')
                v = matrix[w2idx[w]]
                for j in v:
                    out.write('{0:.7}'.format(j))
                out.write('\n')
            except UnicodeEncodeError:
                pass

def test_embeddings(bi, embedding_file, file_type):
    """
    bi: if true, use a bidirectional lstm, otherwise use a normal lstm
    embedding_file: the word embeddings file
    file_type:      word2vec, glove, tang, bin

    Use averaged word embeddings for each word in a text as features
    for l2 regularized logistiic regression. We test the embeddings
    on 10 benchmarks.


    Stanford Sentiment corpus (Socher et al., 2013)
    OpeNER corpus (Agerri et al., 2016)
    Sentube Corpora (Severyn et al., 2016)
    Semeval 2016 twitter corpus - task A
    

    """

    print('importing vectors...')
    vecs = WordVecs(embedding_file, file_type)
    dim = vecs.vector_size
    lstm_dim=50
    dropout=.3
    train=True

    print('Importing datasets...')
    st_fine = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            None,
                                            one_hot=True,
                                            binary=False,
                                            rep=words)
    

    st_binary = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            None,
                                            one_hot=True,
                                            binary=True,
                                            rep=words)

    opener_dataset = General_Dataset('datasets/opener',
                                     None,
                                     one_hot=True,
                                     rep=words)


    sentube_auto_dataset = General_Dataset('datasets/SenTube/auto',
                                           None, rep=words,
                                           binary=True,
                                           one_hot=True)

    sentube_tablets_dataset = General_Dataset('datasets/SenTube/tablets',
                                              None, rep=words,
                                              binary=True,
                                              one_hot=True)

    semeval_dataset = Semeval_Dataset('datasets/semeval',
                                                None, rep=words,
                                                one_hot=True)
    
    datasets = [st_fine, st_binary, opener_dataset, 
                sentube_auto_dataset, sentube_tablets_dataset, semeval_dataset]

    names = ['sst_fine', 'sst_binary', 'opener',
             'sentube_auto', 'sentube_tablets', 'semeval']

    # Collect results here
    results = []
    std_devs = []


    for name, dataset in zip(names, datasets):
        print('Testing on {0}...'.format(name))
        
        max_length = 0
        vocab = {}
        for sent in list(dataset._Xtrain) + list(dataset._Xdev) + list(dataset._Xtest):
            if len(sent) > max_length:
                max_length = len(sent)
            for w in sent:
                if w not in vocab:
                    vocab[w] = 1
                else:
                    vocab[w] += 1

        wordvecs = {}
        for w in vecs._w2idx.keys():
            if w in vocab:
                wordvecs[w] = vecs[w]

        add_unknown_words(wordvecs, vocab, min_df=1, dim=dim)
        W, word_idx_map = get_W(wordvecs, dim=dim)

        print('Converting and Padding dataset...')

        dataset = convert_dataset(dataset, word_idx_map, max_length)

        
        output_dim = dataset._ytest.shape[1]

        """
        Get best Dev params
        ===========================================================
        """
        if bi:
            dev_params_file = 'dev_params/'+str(W.shape[1])+'_bilstm.dev.txt'
        else:
            dev_params_file = 'dev_params/'+str(W.shape[1])+'_lstm.dev.txt'
        best_dim, best_dropout, best_epoch, best_f1 = get_dev_params(name, dev_params_file, bi,
                   dataset._Xtrain, dataset._ytrain, dataset._Xdev, dataset._ydev, wordvecs)

        

        """
        Test model 5 times and get averages and std dev.
        """
        print('Running 5 runs to get average and standard deviations')
        dataset_results = []
        for i, it in enumerate(range(5)):
            np.random.seed()
            print(i+1)

            if bi:
                clf = create_BiLSTM(wordvecs, best_dim, output_dim, best_dropout, weights=W, train=train)
                checkpoint = ModelCheckpoint('models/bilstm/' + name +'/run'+ str(i+1)+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
            else:
                checkpoint = ModelCheckpoint('models/lstm/' + name + '/run'+ str(i+1)+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
                clf = create_LSTM(wordvecs, best_dim, output_dim, best_dropout, weights=W, train=train)
                
            h = clf.fit(dataset._Xtrain, dataset._ytrain, validation_data=[dataset._Xdev, dataset._ydev],
                        epochs=best_epoch, verbose=1, callbacks=[checkpoint])

            if bi:
                base_dir = 'models/bilstm/'+ name +'/run'+ str(i+1)
                weights = os.listdir(base_dir)
            else:
                base_dir = 'models/lstm/' + name + '/run'+str(i+1)
                weights = os.listdir(base_dir)
            
            best_val = 0
            best_weights = ''
            for weight in weights:
                val_acc = re.sub('weights.[0-9]*-', '', weight)
                val_acc = re.sub('.hdf5', '', val_acc)
                val_acc = float(val_acc)
                if val_acc > best_val:
                    best_val = val_acc
                    best_weights = weight

            clf = load_model(os.path.join(base_dir, best_weights))
            pred = clf.predict(dataset._Xtest, verbose=1)
            classes = clf.predict_classes(dataset._Xtest, verbose=1)
            if bi:
                prediction_file = 'predictions/bilstm/' + name + '/run' + str(i+1) + '/pred.txt'
                w2idx_file = 'predictions/bilstm/' + name + '/w2idx.pkl'
            else:
                prediction_file = 'predictions/lstm/' + name + '/run' + str(i+1) + '/pred.txt'
                w2idx_file = 'predictions/lstm/' + name + '/w2idx.pkl'
            print_prediction(prediction_file, classes)
            with open(w2idx_file, 'wb') as out:
                pickle.dump(word_idx_map, out)
            
            labels = sorted(set(dataset._ytrain.argmax(1)))
            if len(labels) == 2:
                average = 'binary'
            else:
                average = 'micro'
            mm = MyMetrics(dataset._ytest, pred, labels=labels, average=average)
            acc, precision, recall, micro_f1 = mm.get_scores()
            dataset_results.append([acc, precision, recall, micro_f1])



        # Get the average and std deviation over 10 runs with 10 random seeds    
        dataset_results = np.array(dataset_results)
        ave_results = dataset_results.mean(axis=0) 
        std_results = dataset_results.std(axis=0)
        print(u'acc: {0:.3f} \u00B1{1:.3f}'.format(ave_results[0], std_results[0]))
        print(u'prec: {0:.3f} \u00B1{1:.3f}'.format(ave_results[1], std_results[1]))
        print(u'recall: {0:.3f} \u00B1{1:.3f}'.format(ave_results[2], std_results[2]))
        print(u'f1: {0:.3f} \u00B1{1:.3f}'.format(ave_results[3], std_results[3]))
        
        results.append(ave_results)
        std_devs.append(std_results)

    results.append(list(np.array(results).mean(axis=0)))
    std_devs.append(list(np.array(std_devs).mean(axis=0)))
    names.append('overall')
    
    return names, results, std_devs, dim


def print_results(bi, file, out_file, file_type):

    names, results, std_devs, dim = test_embeddings(bi, file, file_type)

    rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
    table_data = [[name] + result for name, result in zip(names, rr)]
    table = tabulate.tabulate(table_data, headers=['dataset', 'acc', 'prec', 'rec', 'f1'], tablefmt='simple', floatfmt='.3f')

    if out_file:
        with open(out_file, 'a') as f:
            f.write('\n')
            if bi:
                f.write('+++Bidirectional LSTM+++\n')
            else:
                f.write('+++LSTM+++\n')
            f.write(table)
            f.write('\n')
    else:
        print()
        if bi:
            print('Bidirectional LSTM')
        else:
            print('LSTM')
        print(table)
        
def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-bi', default=False, type=bool)
    parser.add_argument('-emb', help='location of embeddings', 
        default='embeddings/wikipedia-sg-50-window10-sample1e-4-negative5.txt')
    parser.add_argument('-file_type', help='glove style embeddings or word2vec style: default is w2v',
        default='word2vec')
    parser.add_argument('-output', help='output file for results', default='./results.txt')
    parser.add_argument('-printout', help='instead of printing to file, print to sysout',
                        type=bool, default=False)

    args = vars(parser.parse_args())
    bi = args['bi']
    embedding_file = args['emb']
    file_type = args['file_type']
    output = args['output']
    printout = args['printout']

    print('testing on %s' % embedding_file)

    if printout:
        print_results(bi, embedding_file, None, file_type)
    else:
        print_results(bi, embedding_file, output, file_type)


if __name__ == '__main__':

    args = sys.argv
    main(args)



