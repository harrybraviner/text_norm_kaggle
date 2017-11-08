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
    output_length = 15 # FIXME - this should be variable based on the output length

    # Define the RNN cells for the input
    with tf.variable_scope('input_net') as scope:
        rnn_input_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
        rnn_input_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
        rnn_input_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_input_cell_1, rnn_input_cell_2])

        ## Setup tensors for the input sequence

        # Index of character (or <STOP> or <RARE> tokens)
        input_ix = tf.placeholder(tf.int32, [batch_size, input_length])
        embedding_matrix = tf.Variable(tf.truncated_normal([train.num_input_chars, embedding_size], mean = 0.1, stddev = 0.02))
        
        input_dense = tf.nn.embedding_lookup(embedding_matrix, input_ix) # Dense encoding of the characters
        # One-hot encoding of the character if it's a 'non-vanilla' char
        input_oh = tf.placeholder(tf.float32, [batch_size, input_length, train.max_nv_chars])

        input_x = tf.concat([input_dense, input_oh], axis = 2)

        # Take input sequence and produce the 'final state' that will be used to generate the output tensor.
        # We throw away the 'outputs' from this RNN since it's just collecting information into a state.
        _, final_state = tf.nn.dynamic_rnn(cell   = rnn_input_multi_cell,
                                           inputs = input_x,
                                           dtype  = tf.float32)

    # Different net for output compared to input
    with tf.variable_scope('output_net') as scope:

        ## Setup tensors for the output sequence
        output_y = tf.placeholder(tf.float32, [output_length, train.num_output_chars])

        rnn_output_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
        rnn_output_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
        rnn_output_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_output_cell_1, rnn_output_cell_2])

        # FIXME - Maybe consider replacing this with the previous character output (or a special <START> token)?
        rnn_output_zeros = tf.zeros(shape = [batch_size, output_length, 0], dtype = tf.float32)    # Used because we need some input

        output_y_hat_logits, _ = tf.nn.dynamic_rnn(cell = rnn_output_multi_cell,
                                                   inputs = rnn_output_zeros,
                                                   initial_state = final_state,
                                                   dtype = tf.float32)



    

    raise NotImplementedError

if __name__ == '__main__':
    main()
