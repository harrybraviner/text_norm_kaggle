#! /usr/bin/python3

import numpy as np
import tensorflow as tf
import dataset

def main():

    train = dataset.TrainingDataset(line_limit = 1000)

    # Set up the RNN
    # Parameters
    layer_width_1 = 16
    layer_width_2 = 16
    embedding_size = 16 # Embedding for characters
    batch_size = 1  # See README for why I'm constrained to batch sizes of 1 at present
    max_input_length = 21 # Includes the <STOP> token at the end
    max_output_length = 31  # Includes the <STOP> token at the end

    # Define the RNN cells for the input
    with tf.variable_scope('input_net') as scope:
        rnn_input_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
        rnn_input_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
        rnn_input_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_input_cell_1, rnn_input_cell_2])

        ## Setup tensors for the input sequence

        input_length = tf.placeholder(tf.int32, [None])

        # Index of character (or <STOP> or <RARE> tokens)
        input_ix = tf.placeholder(tf.int32, [None, max_input_length])
        embedding_matrix = tf.Variable(tf.truncated_normal([train.num_input_chars, embedding_size], mean = 0.1, stddev = 0.02))
        
        input_dense = tf.nn.embedding_lookup(embedding_matrix, input_ix) # Dense encoding of the characters
        # One-hot encoding of the character if it's a 'non-vanilla' char
        input_oh = tf.placeholder(tf.float32, [None, max_input_length, train.max_nv_chars])

        input_x = tf.concat([input_dense, input_oh], axis = 2)

        # Take input sequence and produce the 'final state' that will be used to generate the output tensor.
        # We throw away the 'outputs' from this RNN since it's just collecting information into a state.
        _, final_state = tf.nn.dynamic_rnn(cell            = rnn_input_multi_cell,
                                           inputs          = input_x,
                                           sequence_length = input_length,
                                           dtype           = tf.float32)

    # Different net for generating output than for absorbing input
    with tf.variable_scope('output_net') as scope:

        ## Setup tensors for the output sequence
        output_length = tf.placeholder(tf.int32, [None])
        output_y = tf.placeholder(tf.float32, [None, max_output_length, train.num_output_chars])  # Correct output sequence, with <STOP> token

        rnn_output_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
        rnn_output_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
        rnn_output_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_output_cell_1, rnn_output_cell_2])

        output_layer_w = tf.Variable(tf.truncated_normal([layer_width_2, train.num_output_chars], mean = 0.1, stddev = 0.02))
        output_layer_b = tf.Variable(tf.zeros([train.num_output_chars]))

        start_token_oh = tf.one_hot([train.start_output_token_ix], depth = train.start_output_token_ix + 1)
        start_token_oh = tf.reshape(start_token_oh, shape=[-1, -1, train.start_output_token_ix + 1])
        # Take the output characters, ditch the <STOP> token, and append a 0 in the <START> token slot
        input_characters = tf.concat([output_y[:, :-1, :], tf.zeros_like(output_y[:, :-1, 0:1], dtype = tf.float32)], axis = 2)
        # Put the <START> token at the beginning
        rnn_output_input = tf.concat([start_token_oh, input_characters], axis = 1)

        layer_2_outputs, _ = tf.nn.dynamic_rnn(cell   = rnn_output_multi_cell,
                                                   inputs = rnn_output_input,
                                                   initial_state = final_state,
                                                   sequence_length = output_length,
                                                   dtype = tf.float32)

        output_y_hat_logits = tf.tensordot(layer_2_outputs, output_layer_w, 1) + output_layer_b

    # Loss measurement
    # We have examples of different sizes in the batch.
    # We want to equally weight examples, rather than equally weighting characters of output.
    flattened_y = tf.reshape(output_y, [-1, train.num_output_chars])
    flattened_y_hat_logits = tf.reshape(output_y_hat_logits, [-1, train.num_output_chars])
    y_hat_weights = tf.reshape(tf.cast(output_length, dtype=tf.float32) * tf.reduce_max(output_y, axis=2), [-1])


    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels = flattened_y,
                                                         logits = flattened_y_hat_logits,
                                                         weights = y_hat_weights)
    
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

    def train_one_batch():

        return 0

    raise NotImplementedError

if __name__ == '__main__':
    main()
