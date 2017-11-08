#! /usr/bin/python3

import numpy as np
import tensorflow as tf
import data_cleaning

def main():

    train = data_cleaning.TrainingDataset(line_limit = 1000)

    # Set up the RNN
    # Parameters
    layer_width_1 = 16
    layer_width_2 = 16
    embedding_size = 16 # Embedding for characters
    batch_size = 1  # See README for why I'm constrained to batch sizes of 1 at present
    input_length = 10 # FIXME - this should be variable based on the input length

    # Define the RNN cells for the input
    rnn_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
    rnn_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
    rnn_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell_1, rnn_cell_2])

    ## Setup tensors for the input sequence

    # Index of character (or <STOP> or <RARE> tokens)
    input_ix = tf.placeholder(tf.int32, [batch_size, input_length])
    embedding_matrix = tf.Variable(tf.truncated_normal([train.num_input_chars, embedding_size], mean = 0.1, stddev = 0.02))
    
    input_dense = tf.nn.embedding_lookup(embedding_matrix, input_ix) # Dense encoding of the characters
    # One-hot encoding of the character if it's a 'non-vanilla' char
    input_oh = tf.placeholder(tf.float32, [batch_size, input_length, train.max_nv_chars])

    input_x = tf.concat([input_dense, input_oh], axis = 2)

    

    raise NotImplementedError

if __name__ == '__main__':
    main()
