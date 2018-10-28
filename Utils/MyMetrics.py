from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

class MyMetrics:
    """options:

       average: None - for multiclass
                'binary' - for binary (pos/neg)
                'micro' -
                'macro' -
    """
    
    def __init__(self, y_true, y_pred, one_hot=True, label_names=None,
                 labels=None, pos_label=1, average=None):
        if one_hot==True:
            self._y_true = self.argmax(y_true)
            self._y_pred = self.argmax(y_pred)
        else:
            self._y_true = y_true
            self._y_pred = y_pred
        self._label_names = label_names
        self._labels = labels
        self._pos_label = pos_label
        self._average =  average

    def argmax(self, y_batch):
        return [np.argmax(y, axis=0) for y in y_batch]
    
    def precision(self):
        return precision_score(self._y_true, self._y_pred, self._labels,
                               self._pos_label, self._average)

    def recall(self):
        return recall_score(self._y_true, self._y_pred, self._labels,
                               self._pos_label, self._average)
    
    def f1(self):
        return f1_score(self._y_true, self._y_pred, self._labels,
                               self._pos_label, self._average)

    def accuracy(self):
        return accuracy_score(self._y_true, self._y_pred)
    
    def get_scores(self):
        if self._average == None:
            precision = self.get_3_decimals(self.precision())
            recall = self.get_3_decimals(self.recall())
            f1 = self.get_3_decimals(self.f1())
        else:
            precision = self.precision()
            recall = self.recall()
            f1 = self.f1()
        acc = self.accuracy()
        return [acc, precision, recall, f1]
    
    def get_3_decimals(self, X):
        return [int(x * 1000 + .5) / 1000 for x in X]

    def print_metrics(self):
        row_format = "{:>15}" * (len(self._labels)+1)
        acc_row_format = "{:>15}" * 2
        metrics = ["precision", "recall", "f1", "accuracy"]
        scores = self.get_scores()
        scores.append([(int(self.accuracy() * 100 +.5) / 100), '-','-','-'])
        
        if self._label_names == None:
            print(row_format.format("", * [str(label) for label in self._labels]))
        else:
            print(row_format.format("", * self._label_names))
            
        for metric, row in zip(metrics, scores):
                print(row_format.format(metric, *row))
