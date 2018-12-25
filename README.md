# SSAN-self-attention-sentiment-analysis-classification
Code for the paper "Self-Attention: A Better Building Block for Sentiment Analysis Neural Network Classifiers": http://aclweb.org/anthology/W18-6219, https://arxiv.org/abs/1812.07860 . This paper was published in WASSA 2018 (9th Workshop on Computational Approaches to Subjectivity, Sentiment &amp; Social Media Analysis),  held in conjuction with EMNLP 2018.

Contact: aambarts@sfu.ca

The paper builds upon the work of the paper "Assessing State-of-the-art Sentiment Models on State-of-the-art Sentiment Datasets", Barnes et. al. This repository is a fork of their implementation for the said paper: https://github.com/jbarnesspain/sota_sentiment.

## Abstract
Sentiment Analysis has seen much progress in the past two decades. For the past few years, neural network approaches, primarily RNNs and CNNs, have been the most successful for this task. Recently, a new category of neural networks, self-attention networks (SANs), have been created which utilizes the attention mechanism as the basic building block. Self-attention networks have been shown to be effective for sequence modeling tasks, while having no recurrence or convolutions. In this work we explore the effectiveness of the SANs for sentiment analysis. We demonstrate that SANs are superior in performance to their RNN and CNN counterparts by comparing their classification accuracy on six datasets as well as their model characteristics such as training speed and memory consumption. Finally, we explore the effects of various SAN modifications such as multi-head attention as well as two methods of incorporating sequence position information into SANs.

## Run Self-Attention models
To run the work we've done, simply unzip the google word embeddings in the /embeddings folder (or use your own) and run ```python san.py -emb embeddings/google.txt```.
To change which self-attention architecture that was discussed in the paper you'd like to, see the hparams dictionary object in san.py. Using the values in that dictionary you can configure the san.py script to run SSAN, Transfore Encoder, RPR or PE positional information technques, etc. 
To run the baseline models, follow the instructions from: https://github.com/jbarnesspain/sota_sentiment

## Word Embeddings
Other word embeddings from Barnes et. al can be found: ([available here](http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/sota-sentiment.html))

## Datasets
1. [Stanford Sentiment Treebank](http://aclweb.org/anthology/D/D13/D13-1170.pdf) - fine-grained
2. [Stanford Sentiment Treebank](http://aclweb.org/anthology/D/D13/D13-1170.pdf) - binary
3. [OpeNER](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/4891)
4. [SenTube Auto](https://ikernels-portal.disi.unitn.it/projects/sentube/)
5. [SenTube Tablets](https://ikernels-portal.disi.unitn.it/projects/sentube/)
6. [SemEval 2013 Task 2](https://www.cs.york.ac.uk/semeval-2013/task2.html)


### Reference
```
@inproceedings{Ambartsoumian2017,
  author    = {Ambartsoumian, Artaches and Popowich, Fred},
  title     = {Self-Attention: A Better Building Block for Sentiment Analysis Neural Network Classifiers},
  booktitle = {Proceedings of the 9th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
  month     = {November},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
}
```
