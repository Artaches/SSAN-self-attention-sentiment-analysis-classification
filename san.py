from keras.preprocessing.sequence import pad_sequences
import argparse
import tabulate
from Utils.WordVecs import *
from Utils.MyMetrics import *
from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *

import tensorflow as tf
from tensorflow.contrib.training import HParams
import numpy as np
import random
import os
import sys


from datetime import datetime
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')     # Print time for logging.


# To run:
# python san.py -emb embeddings/google.txt


# Dictionary that is used to store all of the hyper parameters.
hp = HParams(
    # ------------------------------------ Dataset preprocessing ------------------------------------------------------------------------------
    print_train_loss_period=10,
    test_on_train_period=100,               # Calculate the accuracy on the training set every this many training iterations
    run_on_val_and_test_every=100,

    batch_size=128,
    num_training_iterations=50000,
    # num_training_iterations=250,
    run_exps_amount=5,
    inference_batch_sizes=256,              # Batch size to use to calculate the accuarcy on train, val, and test sets.

# ------------------------------------ Model Config -------------------------------------------------------------------------------------------
    # d_model dimensionality
    model_dim=300,
    sentence_representation_dim=300,

    # SSAN
    num_layers=1,
    self_attention_heads=1,
    qkv_projections_bias_and_activation=True,
    self_attention_sublayer_bias_and_activation=True,
    ffnn_sublayer=False,
    self_attention_sublayer_residual_and_norm=False,
    ffnn_sublayer_residual_and_norm=False,
    include_positional_encoding=False,                      # PE
    positional_encoding_denominator_term=10000,
    use_relative_positions=True,                            # RPR
    max_relative_positions=10,                              # clipping distance


    # # Transformer Encoder
    # num_layers=2,
    # self_attention_heads=8,
    # qkv_projections_bias_and_activation=False,              # Linear projections instead of FFNN
    # self_attention_sublayer_bias_and_activation=False,
    # ffnn_sublayer=True,
    # self_attention_sublayer_residual_and_norm=True,
    # ffnn_sublayer_residual_and_norm=True,
    # include_positional_encoding=True,                       # PE
    # positional_encoding_denominator_term=10000,
    # use_relative_positions=False,                           # RPR
    # max_relative_positions=10,

# ------------------------------------ Regularization Techniques ------------------------------------------------------------------------------
    dropout_keep=0.7,
    test_dropout_keep=1.0,

    input_emb_apply_dropout=True,
    self_attention_sublayer_dropout=True,                # Only enabled if concat ffnn_sublayer is enabled.
    ffnn_sublayer_dropout=True,
    sentence_representation_dropout=True,

# ------------------------------------ Optimizer -------------------------------------------------------------------------------------------
    optimizer="ADADELTA",
    learning_rate=0.1,              # Good for ADADELTA

    # optimizer="ADAM",
    # learning_rate=1e-4,             # Good for Adam
    
    adam_beta_1=0.95,              # Acts as rho when adadelta is used.
    adam_beta_2=0.98,
)

def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(hp.positional_encoding_denominator_term, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([1, sentence_length, dim]), dtype=dtype)

def generate_relative_positions_matrix(length, max_relative_position):
  """
    From tensor2tensor: https://github.com/tensorflow/tensor2tensor
    Generates matrix of relative positions between inputs. [length, length] positional info matrix.
  """
  range_vec = tf.range(length)
  range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
  distance_mat = range_mat - tf.transpose(range_mat)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position, max_relative_position)  # clips all values to range (-k, k)
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat

def generate_relative_positions_embeddings(length, depth, max_relative_position, name):
  """Generates tensor of size [length, length, depth]."""
  with tf.variable_scope(name):
    relative_positions_matrix = generate_relative_positions_matrix(length, max_relative_position)
    vocab_size = max_relative_position * 2 + 1
    embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])   # Create a randomized values matrix
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)     # Fill every position with same randomized values of size depth
    return embeddings

def project_qkv(input_sequnce, output_dim, use_bias_and_activation=True):
    if use_bias_and_activation:
        return tf.layers.dense(input_sequnce, output_dim, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
    else:
        return tf.layers.dense(input_sequnce, output_dim, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.glorot_normal_initializer())

def split_heads(q, k, v):
    def split_last_dimension_then_transpose(tensor, num_heads, dim):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]
    qs = split_last_dimension_then_transpose(q, hp.self_attention_heads, hp.model_dim)
    ks = split_last_dimension_then_transpose(k, hp.self_attention_heads, hp.model_dim)
    vs = split_last_dimension_then_transpose(v, hp.self_attention_heads, hp.model_dim)
    return qs, ks, vs

def scaled_dot_product(qs, ks, vs):
    '''
      Regular self-attention using scaled dot-product compatibility function.
    '''
    key_dim_per_head = hp.model_dim // hp.self_attention_heads
    logits = tf.matmul(qs, ks, transpose_b=True)    # Dot products, [batch_size, max_seq_len, max_seq_len]
    logits = logits / (key_dim_per_head**0.5)     # Scale the dot product
    attention_weights = tf.nn.softmax(logits)
    return tf.matmul(attention_weights, vs)

def relative_attention_inner(x, y, z, transpose):
    """
    From tensor2tensor: https://github.com/tensorflow/tensor2tensor

    Relative position-aware dot-product attention inner calculation.
    This batches matrix multiply calculations to avoid unnecessary broadcasting.
    Args:
      x: Tensor with shape [batch_size, heads, max_seq, max_seq or dim].  max_seq if dot-product of weights=q*k, dim if output=weights*v
      y: Tensor with shape [batch_size, heads, max_seq, dim].
      z: Tensor with shape [max_seq, max_seq, dim].
      transpose: Whether to transpose inner matrices of y and z. Should be true if last dimension of x is dim, not max_seq.
    Returns:
      A Tensor with shape [batch_size, heads, max_seq, max_seq or dim].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    max_seq = tf.shape(x)[2]
    
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)                          # xy_matmul is [batch_size, heads, max_seq, max_seq or dim]
    x_t = tf.transpose(x, [2, 0, 1, 3])                                         # x_t is [max_seq, batch_size, heads, max_seq or dim]
    x_t_r = tf.reshape(x_t, [max_seq, heads * batch_size, -1])                  # x_t_r is [max_seq, batch_size * heads, max_seq or dim]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)                    # x_tz_matmul is [max_seq, batch_size * heads, max_seq or dim]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [max_seq, batch_size, heads, -1])   # x_tz_matmul_r is [max_seq, batch_size, heads, max_seq or dim]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])                 # x_tz_matmul_r_t is [batch_size, heads, max_seq, max_seq or dim]
    return xy_matmul + x_tz_matmul_r_t

def dot_product_attention_relative(q, k, v, name=None):
    """
    From tensor2tensor: https://github.com/tensorflow/tensor2tensor

    Calculate relative position-aware dot-product self-attention.
    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.
    Args:
      q: a Tensor with shape [batch, heads, max_seq, dim].
      k: a Tensor with shape [batch, heads, max_seq, dim].
      v: a Tensor with shape [batch, heads, max_seq, dim].
      max_relative_position: an integer specifying the maximum distance between
          inputs that unique position embeddings should be learned for.
      dropout_keep_prob: a floating point number.
      name: an optional string.
    Returns:
      A Tensor. [batch, head, max_seq, dim]
    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not hp.max_relative_positions:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                       "relative self attention." % (hp.max_relative_positions))

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

    # Use separate embeddings suitable for keys and values.
    dim_size = q.get_shape().as_list()[3]
    max_seq = q.get_shape().as_list()[2]

    # initialize the relative position matrices for this layer, if non were passed. Thus not sharing weights between stacks.
    relativePositionsForKeys = generate_relative_positions_embeddings(
            max_seq, dim_size, hp.max_relative_positions, "relative_positions_keys")
    relativePositionsForValues = generate_relative_positions_embeddings(
            max_seq, dim_size, hp.max_relative_positions, "relative_positions_values")

    # Compute self attention considering the relative position embeddings.
    logits = relative_attention_inner(q, k, relativePositionsForKeys, True)
    logits = logits / (dim_size**(0.5))     # Scale the dot product
    weights = tf.nn.softmax(logits, name="attention_weights")     #[bs, ml, ml]   bs = batch_size, ml = max_sentence_length

    return relative_attention_inner(weights, v, relativePositionsForValues, False)

# Concatenate all the heads.
def concatenate_heads(outputs):
    def transpose_then_concat_last_two_dimenstion(tensor):
        tensor = tf.transpose(tensor, [0, 2, 1, 3])                         # [batch_size, max_seq_len, num_heads, dim]
        t_shape = tensor.get_shape().as_list()
        num_heads, dim = t_shape[-2:]
        return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])
    return transpose_then_concat_last_two_dimenstion(outputs)               # [batch_size, max_seq_len, dim]

def multi_head_attention(input_sequence, dropout_keep_prob_tensor):
    '''
        Returns a self-attention layer, configured as to the parameters in the global hparams dictionary.
    '''

    # make sure the input word embedding dimension divides by the number of desired heads.
    assert hp.model_dim % hp.self_attention_heads == 0
    qkv_dim = hp.model_dim / hp.self_attention_heads

    # Construct the Q, K, V matrices
    q = project_qkv(input_sequence, qkv_dim, hp.qkv_projections_bias_and_activation)
    k = project_qkv(input_sequence, qkv_dim, hp.qkv_projections_bias_and_activation)
    v = project_qkv(input_sequence, qkv_dim, hp.qkv_projections_bias_and_activation)
    qs, ks, vs = split_heads(q, k, v)

    if hp.use_relative_positions:
      outputs = dot_product_attention_relative(qs, ks, vs)
    else:
      outputs = scaled_dot_product(qs, ks, vs)

    san_output = concatenate_heads(outputs)

    if hp.self_attention_sublayer_bias_and_activation:
        san_output = tf.layers.dense(san_output, hp.model_dim, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
    else:
        san_output = tf.layers.dense(san_output, hp.model_dim, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.glorot_normal_initializer())

    if hp.self_attention_sublayer_dropout:
        san_output = tf.nn.dropout(san_output, keep_prob=dropout_keep_prob_tensor)          # ignore some input info to regularize the model

    return san_output



def encoder_layer(input_sequence, dropout_keep_prob_tensor):
    self_attention_layer = multi_head_attention(input_sequence, dropout_keep_prob_tensor)
    
    if hp.self_attention_sublayer_residual_and_norm:
        self_attention_layer = tf.add(self_attention_layer, input_sequence)
        self_attention_layer = tf.contrib.layers.layer_norm(self_attention_layer)

    # Add the 2-layer feed-forward with residual connections and layer normalization. Transformer uses it.
    if hp.ffnn_sublayer:
        ffnn_sublayer_output = tf.layers.dense(self_attention_layer, hp.model_dim, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
        ffnn_sublayer_output = tf.layers.dense(ffnn_sublayer_output, hp.model_dim, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
        if hp.ffnn_sublayer_dropout:
             ffnn_sublayer_output = tf.nn.dropout(ffnn_sublayer_output, keep_prob=dropout_keep_prob_tensor)          # ignore some input info to regularize the model
        ffnn_sublayer_output = tf.add(ffnn_sublayer_output, self_attention_layer)
        encoder_layer_output = tf.contrib.layers.layer_norm(ffnn_sublayer_output)
    else:
        encoder_layer_output = self_attention_layer

    return encoder_layer_output


def transformerClassifier(x_tensor, output_dim, wordIndxToVec_tensor, dropoutKeep_tensor, max_sentence_length):
    with tf.variable_scope("Embedding_Layer"):
        emb = tf.nn.embedding_lookup(wordIndxToVec_tensor, x_tensor)

    # Add positional encodings to the embeddings we feed to the encoder.
    if hp.include_positional_encoding:
        with tf.variable_scope("Add_Position_Encoding"):
            posEnc = positional_encoding(hp.model_dim, max_sentence_length)
            emb = tf.add(emb, posEnc, name="Add_Positional_Encoding")
            
    if hp.input_emb_apply_dropout:
        with tf.variable_scope("Input_Embeddings_Dropout"):
            emb = tf.nn.dropout(emb, keep_prob=dropoutKeep_tensor)          # ignore some input info to regularize the model

    for i in range(1, hp.num_layers + 1):
        with tf.variable_scope("Stack-Layer-{0}".format(i)):
            encoder_output = encoder_layer(emb, dropout_keep_prob_tensor=dropoutKeep_tensor)
            emb = encoder_output

    # Simply average the final sequence position representations to create a fixed size "sentence representation".
    sentence_representation = tf.reduce_mean(encoder_output, axis=1)    # [batch_size, model_dim]

    with tf.variable_scope("Sentence_Representation_And_Output"):
        sentence_representation = tf.layers.dense(sentence_representation, hp.model_dim, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
        if hp.sentence_representation_dropout:
            sentence_representation = tf.nn.dropout(sentence_representation, keep_prob=dropoutKeep_tensor)          # ignore some input info to regularize the model

        prediction_logits = tf.layers.dense(sentence_representation, output_dim, activation=None, use_bias=False, kernel_initializer=tf.glorot_normal_initializer())

    return prediction_logits


def predictInBatches(sess, predictTfOp, xData, yData, x_tensor, y_tensor, feed_dict, batch_size=32):

    datasetSize = xData.shape[0]
    currPos = 0
    predictions = np.empty_like(yData, dtype=np.float32)
    while currPos < datasetSize - 1:
        if currPos + batch_size < datasetSize - 1:
            nextPos = currPos + batch_size
        else:
            nextPos = datasetSize

        curr_feed_dict = dict(feed_dict)
        curr_feed_dict[x_tensor] = xData[currPos:nextPos]
        curr_feed_dict[y_tensor] = yData[currPos:nextPos]
        pred_val = sess.run(predictTfOp, curr_feed_dict)

        predictions[currPos:nextPos] = pred_val
        currPos = nextPos
    
    return predictions


def printNumberOfParams():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        # print("shape:", shape, "len(shape):", len(shape))
        variable_parameters = 1
        for dim in shape:
            # print("\t", dim)
            variable_parameters *= dim.value
        # print("\tparams:", variable_parameters)
        total_parameters += variable_parameters
    print("\nNumber of trainable parameters in the network:", total_parameters)



def createAndTrainTransformer(datasets, wordIndxToVecMatrix, wordIndxToVec_tensor, output_dim, datasetName, max_sentence_length):

    x_tensor = tf.placeholder(tf.int32, [None, max_sentence_length], name="input_sentence")     # [batch_size, num_steps, input_dim]
    y_tensor = tf.placeholder(tf.float32, [None, output_dim], name="targets")                  # [batch_size, target_dim] We're predicting only 1 number at the end...
    dropoutKeep_tensor = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_probability")
    
    # Model output & class predictions.
    output = transformerClassifier(x_tensor, output_dim, wordIndxToVec_tensor, dropoutKeep_tensor, max_sentence_length)
    softmax = tf.nn.softmax(output, name="softmax")
    prediction = tf.one_hot(tf.nn.top_k(softmax).indices, tf.shape(softmax)[-1], axis=-1)
    prediction = tf.squeeze(prediction)     # Removes dimensions of 1 from the tensor: [n,1,k] -> [n,k]

    # Cost function
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_tensor))
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.argmax(y_tensor, axis=1)))

    if hp.optimizer == "ADAM":
        optimizer = tf.train.AdamOptimizer(learning_rate=hp.learning_rate, beta1=hp.adam_beta_1, beta2=hp.adam_beta_2, epsilon=1e-9).minimize(cost)
    elif hp.optimizer == "ADADELTA":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=hp.learning_rate, rho=hp.adam_beta_1).minimize(cost)

    printNumberOfParams()       # Print statistics for number of trainable parameters in the model.

    init = tf.global_variables_initializer()
    config = tf.ConfigProto(
        gpu_options={'allow_growth': True},   # Dynamic allocation of GPU memory. Doesn't allocate all of GPU VRAM at initialization.
        allow_soft_placement=True,            # Places model on CPU if some model operations are not supported by GPU
    )
    with tf.Session(config=config) as sess:
        sess.run(init)

        # setup tensorboard writers
        start_time = datetime.now()
        # strTimeStr = start_time.strftime('%Y-%m-%d;%H;%M;%S')

        best_acc_val = 0
        best_mm_val = None
        best_mm_test = None
        topValidationSaves = []


        for iteration in range(1, hp.num_training_iterations):
            iteration_start_time = datetime.now()

            # Construct a training batch by random sampling
            datasetSizeArange = list(np.arange(0, datasets._Xtrain.shape[0]))       # List of indexes for train set
            indices = np.array(random.sample(datasetSizeArange, hp.batch_size))    # get hp.batchSize random index numbers between [0, trainSize]
            X = datasets._Xtrain[indices]   # "fancy" indices
            Y = datasets._ytrain[indices]
            
            sess.run([optimizer], feed_dict={x_tensor: X, y_tensor: Y, wordIndxToVec_tensor: wordIndxToVecMatrix,
                                                    dropoutKeep_tensor: hp.dropout_keep})
            
            # Calculate batch cost and print it.
            if iteration % hp.print_train_loss_period == 0:
                cost_val = sess.run(cost, feed_dict={x_tensor: X, y_tensor: Y, wordIndxToVec_tensor: wordIndxToVecMatrix,
                                                    dropoutKeep_tensor: hp.dropout_keep})
                print("    Iteration #{}, time for iteration = {}, iteration cost = {}".format(
                        iteration, str(datetime.now() - iteration_start_time), str(cost_val)))

            # Calculate accuracy on the entire training set. This helps keep track of overfitting.
            mm_train = None
            if iteration % hp.test_on_train_period == 0:
                labels = sorted(set(datasets._ytrain.argmax(1)))     # argmax of the target label is num of classes since they're integers [0,n]
                if len(labels) == 2:
                    average = 'binary'
                else:
                    average = 'micro'

                # Print Training set Metrics
                xTrain = datasets._Xtrain
                yTrain = datasets._ytrain
                feed_dict = {wordIndxToVec_tensor: wordIndxToVecMatrix, dropoutKeep_tensor: hp.test_dropout_keep}
                pred_train = predictInBatches(sess, prediction, xTrain, yTrain, x_tensor, y_tensor, feed_dict, batch_size=hp.inference_batch_sizes)
    
                mm_train = MyMetrics(yTrain, pred_train, labels=labels, average=average)
                acc_train, precision_train, recall_train, micro_f1_train = mm_train.get_scores()
                print("\n",)
                print("---------------------------------------------------------------")
                print("Training:")
                print("\tacc:", acc_train)
                print("\tprecision:", precision_train)
                print("\trecall:", recall_train)
                print("\tmicro_f1:", micro_f1_train)

            # Calculate accruacy for the validation and test sets.
            if iteration % hp.run_on_val_and_test_every == 0:
                # Print Validation set Metrics
                feed_dict = {wordIndxToVec_tensor: wordIndxToVecMatrix, dropoutKeep_tensor: hp.test_dropout_keep}  # Add The basic crap to the feed_dict
                pred_val = predictInBatches(sess, prediction, datasets._Xdev, datasets._ydev, x_tensor, y_tensor,
                                    feed_dict, batch_size=hp.inference_batch_sizes)
                mm_val = MyMetrics(datasets._ydev, pred_val, labels=labels, average=average)
                acc_val, precision_val, recall_val, micro_f1_val = mm_val.get_scores()
                print("\nValidation:",)
                print("\tacc:", acc_val)
                print("\tprecision:", precision_val)
                print("\trecall:", recall_val)
                print("\tmicro_f1:", micro_f1_val)


                # Print Test set Metrics
                feed_dict = {wordIndxToVec_tensor: wordIndxToVecMatrix, dropoutKeep_tensor: hp.test_dropout_keep}  # Add The basic crap to the feed_dict
                pred_test = predictInBatches(sess, prediction, datasets._Xtest, datasets._ytest, x_tensor, y_tensor,
                                    feed_dict, batch_size=hp.inference_batch_sizes)
                mm_test = MyMetrics(datasets._ytest, pred_test, labels=labels, average=average)
                acc_test, precision_test, recall_test, micro_f1_test = mm_test.get_scores()
                print("Test:",)
                print("\tacc:", acc_test)
                print("\tprecision:", precision_test)
                print("\trecall:", recall_test)
                print("\tmicro_f1:", micro_f1_test)

                print("Time running: " + str(datetime.now() - start_time))

                if acc_val >= best_acc_val:
                    best_mm_val = mm_val
                    best_mm_test = mm_test
                    best_acc_val = acc_val

                    print("---------------------------------------------------------------------------------------------------------------")
                    print("New Best Validation Accuracy!")
                    print("---------------------------------------------------------------------------------------------------------------")
                    print("\n")
                    # modelSaveName = base_model_dir + strTimeStr + "-(data={})-(iteration={})-(val_acc={:.4f})-(test_acc={:.4f}).ckpt".format(datasetName, epoch, acc_val, acc_test)
                    # save_path = saver.save(sess, modelSaveName)

                    if mm_train is not None:
                        topValidationSaves.append((iteration, acc_train, acc_val, acc_test))
                    else:
                        topValidationSaves.append((iteration, None, acc_val, acc_test))

    # print last 20 model "saves", ie, top validation scores.
    numValSavedToPrint = min(len(topValidationSaves), 20)
    print("\n\nTotal Saved Validations:", len(topValidationSaves))
    print("Top {} Validations:".format(numValSavedToPrint))
    for (iteration, train_acc, val_acc, test_acc) in topValidationSaves[-numValSavedToPrint:]:
        if train_acc is not None:
            print("Iteration: {}, Train: {:.4f},\t\t Val: {:.4f}, Test: {:.4f}".format(iteration, train_acc, val_acc, test_acc))
        else:
            print("Iteration: {}, Train: {},\t Val: {:.4f}, Test: {:.4f}".format(iteration, train_acc, val_acc, test_acc))

    clf = prediction    # The output node is our classifier
    return clf, best_mm_val, best_mm_test

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


def idx_sent(sent, w2idx):
    return np.array([w2idx[w] for w in sent])

def write_vecs(matrix, w2idx, outfile):
    vocab = sorted(w2idx.keys())
    with open(outfile, 'w', encoding="utf8") as out:
        for w in vocab:
            try:
                out.write(w + ' ')
                v = matrix[w2idx[w]]
                for j in v:
                    out.write('{0:.7}'.format(j))
                out.write('\n')
            except UnicodeEncodeError:
                pass

def printMetrics(mean_vec, std_vec, dataset_name):
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("Best metric statistics for:", dataset_name)
    print(u'acc: {0:.3f} \u00B1{1:.3f}'.format(mean_vec[0], std_vec[0]))
    print(u'prec: {0:.3f} \u00B1{1:.3f}'.format(mean_vec[1], std_vec[1]))
    print(u'recall: {0:.3f} \u00B1{1:.3f}'.format(mean_vec[2], std_vec[2]))
    print(u'f1: {0:.3f} \u00B1{1:.3f}'.format(mean_vec[3], std_vec[3]))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")

def convert_dataset(dataset, w2idx, datasetName, maxlen=50):
    '''
        Convert the dataset XTrain, Xdev, and Xtest sets to padded vectors of w2v dict indexes.
        Returns: datasets in format: []
    '''
    # init and fill the numpy ndarrays
    dataset._Xtrain = np.array([idx_sent(s, w2idx) for s in dataset._Xtrain])
    dataset._Xdev = np.array([idx_sent(s, w2idx) for s in dataset._Xdev])
    dataset._Xtest = np.array([idx_sent(s, w2idx) for s in dataset._Xtest])

    # Pad "post" since we'll be applying positional encoding to the vectors. Thus, padding needs to be after the sentence is over.
    dataset._Xtrain = pad_sequences(sequences=dataset._Xtrain, maxlen=maxlen, padding="post")
    dataset._Xdev = pad_sequences(sequences=dataset._Xdev, maxlen=maxlen, padding="post")
    dataset._Xtest = pad_sequences(sequences=dataset._Xtest, maxlen=maxlen, padding="post")

    # Phrase level SST dataset. Still run metrics on sentence level Train set.
    if datasetName == 'sst_fine_phrase_level' or datasetName == 'sst_binary_phrase_level':
        dataset._Xtrain_for_testing = np.array([idx_sent(s, w2idx) for s in dataset._Xtrain_for_testing])
        dataset._Xtrain_for_testing = pad_sequences(sequences=dataset._Xtrain_for_testing, maxlen=maxlen, padding="post")

    return dataset

def add_unknown_words(vecs, wordvecs, vocab, min_df=1, dim=50):
    """
    For words that occur at least min_df times, create a separate word vector.
    0.25 is chosen so the unk vectors have approximately the same variance as pretrained ones
    """
    num_word_unknown = 0.0
    num_word_unknown_lower_cased = 0.0
    for word in vocab:
        if word not in wordvecs:
            num_word_unknown += 1
            if word.lower() not in wordvecs:
                num_word_unknown_lower_cased += 1

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

def run_model_on_datasets_with_embeddings(embedding_file, file_type):
    """
    embedding_file: the word embeddings file
    file_type:      word2vec, glove
    """
    print('importing word embedding vectors...')
    vecs = WordVecs(embedding_file, file_type)  # load the word2vec dictionary.
    dim = vecs.vector_size      # dimensionality of the word embeddings

    # For collecting results to return
    results = []
    std_devs = []


    datasetNames = [
                # 'sst_fine',
                # 'sst_binary',
                # 'opener',
                # 'sentube_auto',
                'sentube_tablets',
                'semeval',
                ]

    # train & test the model on every dataset above
    for datasetName in datasetNames:
        # dataset_load_start = datetime.now()
        if datasetName == 'sst_fine':
            dataset = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            None,
                                            one_hot=True,
                                            binary=False,
                                            rep=words)
        elif datasetName == 'sst_binary':
            dataset = Stanford_Sentiment_Dataset('datasets/stanford_sentanalysis',
                                            None,
                                            one_hot=True,
                                            binary=True,
                                            rep=words)
        elif datasetName == 'opener':
            dataset = General_Dataset('datasets/opener',
                                             None,
                                             one_hot=True,
                                             rep=words)
        
        elif datasetName == 'sentube_auto':
            dataset = General_Dataset('datasets/SenTube/auto',
                                                   None, rep=words,
                                                   binary=True,
                                                   one_hot=True)
        elif datasetName == 'sentube_tablets':
            dataset = General_Dataset('datasets/SenTube/tablets',
                                                      None, rep=words,
                                                      binary=True,
                                                      one_hot=True)
        elif datasetName == 'semeval':
            dataset = Semeval_Dataset('datasets/semeval',
                                                        None, rep=words,
                                                        one_hot=True)


        print('Loading & Testing on {}:'.format(datasetName))

        # if hp.lowercase_all_sentences:
        #     for sent in dataset._Xtrain:
        #         for word in sent:
        #             if word != word.lower():
        #                 print("Word has an uppercase character:", word.decode('utf-8'))


        # find out the max length of sentences in the dataset and construct the vocab frequency dict.
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

        # create a dict of words that are in our word2vec embeddings
        # wordvecs: String -> embedding_vec
        wordvecs = {}
        for w in vecs._w2idx.keys():
            if w in vocab:
                wordvecs[w] = vecs[w]

        # Assign random w2v vectors to the unknown words. These are random uniformly distrubuted vectors of size dim.
        add_unknown_words(vecs, wordvecs, vocab, min_df=1, dim=dim)
        W, word_idx_map = get_W(wordvecs, dim=dim)  # Get the w2v index map for out final vocab

        print('Converting dataset to being right padded...')
        dataset = convert_dataset(dataset, word_idx_map, datasetName, max_length)
        output_dim = dataset._ytest.shape[1]

        # Test model hp.run_exps_amount times and get averages and std dev.
        dataset_results = []
        for i in range(1, hp.run_exps_amount + 1):  
            tf.reset_default_graph()  # Clears the current loaded tensorflow graph.

            w2i = tf.Variable(tf.constant(0.0, shape=[W.shape[0], W.shape[1]]),
                trainable=False, name="embedding_table")
            wordIndxToVec_tensor = tf.placeholder(tf.float32, [W.shape[0], W.shape[1]], name="embedding_table")     # [vobab_size x word_embedding_dim]
            w2i.assign(wordIndxToVec_tensor)

            start_time = datetime.now()     # Print time for logging.
            clf, best_mm_val, best_mm_test = createAndTrainTransformer(dataset, W, wordIndxToVec_tensor, output_dim, datasetName, max_length)
            print("Finished run #", i, "Time taken: " + str(datetime.now() - start_time))

            mm = best_mm_test
            acc, precision, recall, micro_f1 = mm.get_scores()
            dataset_results.append([acc, precision, recall, micro_f1])
            if hp.run_exps_amount == 1:
                acc, precision, recall, micro_f1 = mm.get_scores()
                dataset_results.append([acc, precision, recall, micro_f1])  # add twice so the average is the same... avoid running multiple runs this way.

            if hp.run_exps_amount != 1:   # Print the metrics for this run, unless we're running experiment only once.
                this_run_result = []
                this_run_result.append([acc, precision, recall, micro_f1])
                this_run_result.append([acc, precision, recall, micro_f1])
                this_run_result = np.array(this_run_result)
                this_run_ave_results = this_run_result.mean(axis=0) 
                this_run_std_results = this_run_result.std(axis=0)
                printMetrics(this_run_ave_results, this_run_std_results, datasetName)

        # Get the average and std deviation over 10 runs with 10 random seeds    
        dataset_results = np.array(dataset_results)
        ave_results = dataset_results.mean(axis=0)
        std_results = dataset_results.std(axis=0)
        printMetrics(ave_results, std_results, datasetName)
        
        results.append(ave_results)
        std_devs.append(std_results)

    results.append(list(np.array(results).mean(axis=0)))
    std_devs.append(list(np.array(std_devs).mean(axis=0)))
    datasetNames.append('overall')
    
    return datasetNames, results, std_devs, dim


def print_results(file, out_file, file_type):
    '''
        file: embedding file name
        out_file: file name to save the output
        file_type: word embedding type. w2v (word2vec) or glove
    '''

    iteration_start_time = datetime.now()

    # Run the experiments and get the results back
    names, results, std_devs, dim = run_model_on_datasets_with_embeddings(file, file_type)

    print("Time taken for all experiments:", datetime.now() - iteration_start_time)

    # Print the results in a fancy stdout table. And or save the fancy table to results file.
    rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
    table_data = [[name] + result for name, result in zip(names, rr)]
    table = tabulate.tabulate(table_data, headers=['dataset', 'acc', 'prec', 'rec', 'f1'], tablefmt='simple', floatfmt='.3f')

    if out_file:
        with open(out_file, 'a', encoding="utf8") as f:
            f.write('\n')
            f.write("Hparams: {}".format(hp))
            f.write('+++Transformer+++\n')
            f.write(table)
            f.write('\n\n\n')
    else:
        print('\n')
        print('+++Transformer+++\n')
        print(table)
        print('\n')


def main(args):
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('-emb', help='location of embeddings', 
        default='embeddings/wikipedia-sg-50-window10-sample1e-4-negative5.txt')
    parser.add_argument('-file_type', help='glove style embeddings or word2vec style: default is w2v',
        default='word2vec')

    experimentStartTime = datetime.now()
    parser.add_argument('-output', help='output file for results', default="./results/transformer_results-({}).txt".format(experimentStartTime.strftime('%Y-%m-%d;%H-%M-%S')))
    # parser.add_argument('-printout', help='instead of printing to file, print to sysout', type=bool, default=False)
    parser.add_argument('-gpus', help='what gpus to use', default="0")

    print("Sentiment Analysis Exps - Using SAN model.")
    print(experimentStartTime.strftime('%Y-%m-%d %H:%M:%S'), "\n\n")

    args = vars(parser.parse_args())
    embedding_file = args['emb']
    file_type = args['file_type']
    output = args['output']
    # printout = args['printout']
    gpus = args['gpus']


    if gpus is not None:    # select which gpus to use
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    print('testing on %s' % embedding_file)
    print_results(embedding_file, output, file_type)


if __name__ == '__main__':
    args = sys.argv
    main(args)
