import numpy as np


def sum_vecs(sentence, model):
    """Returns the sum of the vectors of the tokens
    in the sentence if they are in the model"""
    sent = np.array(np.zeros((model.vector_size)))
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            pass
    return sent


def ave_vecs(sentence, model):
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += model['the']
    return sent / sent_length


def idx_vecs(sentence, model):
    """Returns a list of vectors of the tokens
    in the sentence if they are in the model."""
    sent = []
    for w in sentence.split():
        try:
            sent.append(model[w])
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent.append(model['of'])
    return sent


def words(sentence, model):
    return sentence.split()


def getMyData(fname, label, model, representation=sum_vecs, encoding='utf8'):
    data = []
    for sent in open(fname, encoding="utf8"):
        data.append((representation(sent, model), label))
    return data

