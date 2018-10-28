import sys
import tabulate
import argparse
import json
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Dense, Embedding, Convolution1D, MaxPooling1D, Flatten, Merge, Input
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.Representations import words as word_reps
from Utils.Semeval_2013_Dataset import *
from Utils.MyMetrics import *


def print_prediction(file, prediction):
    with open(file, 'w') as out:
        for line in prediction:
            out.write(str(line) + '\n')


def get_dev_params(dataset_name, outfile, maxlen,
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

        
        clf = create_cnn(vecs, maxlen, dim, dropout, output_dim)
        h = clf.fit(Xtrain, ytrain, nb_epoch=epoch, verbose=1)
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
    


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def create_cnn(W, max_length, dim=300,
               dropout=.5, output_dim=8):

    # Convolutional model
    filter_sizes=(2,3,4)
    num_filters = 3
   

    graph_in = Input(shape=(max_length, len(W[0])))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                 filter_length=fsz,
                 border_mode='valid',
                 activation='relu',
                 subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
        
    out = Merge(mode='concat')(convs)
    graph = Model(input=graph_in, output=out)

    # Full model
    model = Sequential()
    model.add(Embedding(output_dim=W.shape[1],
                        input_dim=W.shape[0],
                        input_length=max_length, weights=[W],
                        trainable=True))
    model.add(Dropout(dropout))
    model.add(graph)
    model.add(Dense(dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                  metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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
        out.write('{0} {1}\n'.format(len(vocab), matrix.shape[1]))
        for w in vocab:
            out.write(w  + ' ')
            v = matrix[w2idx[w]]
            for j in v:
                out.write('{0:.7} '.format(j))
            out.write('\n')


def test_embeddings(file, file_type):
    print('Importing vecs...')
    #vec_file = sys.argv[1]
    #vec_file = '/home/jeremy/Escritorio/sentiment_retrofitting/embeddings/sswe-u-50.txt'
    vecs = WordVecs(file)

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
                                     vecs,
                                     one_hot=True,
                                     rep=word_reps)

    twitter_dataset = Semeval_Dataset('datasets/twitter',
                                                    vecs._w2idx, rep=word_reps,
                                                    one_hot=True)

    sentube_auto_dataset = General_Dataset('datasets/SenTube/auto',
                                           vecs._w2idx, rep=word_reps,
                                           binary=True,
                                           one_hot=True)

    sentube_tablets_dataset = General_Dataset('datasets/SenTube/tablets',
                                              vecs._w2idx, rep=word_reps,
                                              binary=True,
                                              one_hot=True)

    semeval_dataset = Semeval_Dataset('datasets/semeval',
                                                vecs, rep=words,
                                                one_hot=True)

    datasets = [st_fine, st_binary, opener_dataset,
                sentube_auto_dataset, sentube_tablets_dataset,
                semeval_dataset]

    names = ['sst_fine', 'sst_binary', 'opener',
             'sentube_auto', 'sentube_tablets', 'semeval']


    dim = vecs.vector_size
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

        add_unknown_words(wordvecs, vocab, min_df=1, k=dim)
        W, word_idx_map = get_W(wordvecs, k=dim)

        print('Converting and Padding dataset...')

        dataset = convert_dataset(dataset, word_idx_map, max_length)

        
        output_dim = dataset._ytest.shape[1]


        """
        Get best Dev params
        ===========================================================
        """
        
        dev_params_file = 'dev_params/'+str(W.shape[1])+'_cnn.dev.txt'
        best_dim, best_dropout, best_epoch, best_f1 = get_dev_params(name, dev_params_file, max_length,
                   dataset._Xtrain, dataset._ytrain, dataset._Xdev, dataset._ydev, W)

        

        # Collect results here
        results = []
        std_devs = []

        for i, it in enumerate(range(5)):
            dataset_results = []

            checkpoint = ModelCheckpoint('models/cnn/' + name + '/run'+ str(i+1)+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')            
            clf = create_cnn(W, max_length, dim=best_dim, dropout=best_dropout, output_dim=output_dim)
            
            h = clf.fit(dataset._Xtrain, dataset._ytrain, validation_data=[dataset._Xdev, dataset._ydev],
                        epochs=best_epoch, verbose=1, callbacks=[checkpoint])


            base_dir = 'models/cnn/' + name + '/run'+ str(i+1)
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
            prediction_file = 'predictions/cnn/' + name + '/run' + str(i+1) + '/pred.txt'
            w2idx_file = 'predictions/cnn/' + name + '/w2idx.pkl'
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

    return names, results, std_devs, dim

def print_results(file, out_file, file_type):

    names, results, std_devs, dim = test_embeddings(file, file_type)

    rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
    table_data = [[name] + result for name, result in zip(names, rr)]
    table = tabulate.tabulate(table_data, headers=['dataset', 'acc', 'prec', 'rec', 'f1'], tablefmt='simple', floatfmt='.3f')
    
    if out_file:
        with open(out_file, 'a') as f:
            f.write('\n')
            f.write('+++CNN+++\n')
            f.write(table)
            f.write('\n')
    else:
        print()
        print('CNN')
        print(table)
        
def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-emb', help='location of embeddings', 
        default='embeddings/wikipedia-sg-50-window10-sample1e-4-negative5.txt')
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



