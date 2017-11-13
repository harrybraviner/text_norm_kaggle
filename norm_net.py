#! /usr/bin/python3

import numpy as np
import tensorflow as tf
import dataset
import argparse, os

def main(log_dir, max_batches, mini_batches):

    max_input_length = 20 # Does not include the <STOP> token at the end
    max_output_length = 30  # Includes the <STOP> token at the end

    line_limit = 1000 if mini_batches else None
    train = dataset.TrainingDataset(line_limit = line_limit, max_input_size = max_input_length, max_output_size = max_output_length)

    # Set up the RNN
    # Parameters
    layer_width_1 = 16
    layer_width_2 = 16
    embedding_size = 16 # Embedding for characters
    max_nv_chars = 10   # Max number of distinct non-vanilla characters that a single input may contain

    # Special token indices to input to the encoder and decoder
    rare_ix  = train.num_non_rare_chars
    stop_ix  = train.num_non_rare_chars + 1
    start_ix = train.num_non_rare_chars + 2 # Decoder input only
    stop_output_ix = train.num_vanilla_chars + max_nv_chars

    num_output_tokens = train.num_vanilla_chars + max_nv_chars + 1  # +1 for <STOP> token
    num_decoder_input_tokens = train.num_non_rare_chars + 3 # <RARE>, <STOP> and <START>

    # Embedding matrix for characters
    # The x-axis ordering should be understood as [vanilla chars] [non-vanilla chars] <RARE> <STOP> <START>
    embedding_matrix = tf.Variable(tf.truncated_normal([num_decoder_input_tokens, embedding_size], mean = 0.1, stddev = 0.02))

    with tf.variable_scope('encoder') as scope:

        # Define the RNN cells for the encoder
        rnn_encoder_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
        rnn_encoder_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
        rnn_encoder_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_encoder_cell_1, rnn_encoder_cell_2])

        ## Setup tensors for the input sequence

        input_length = tf.placeholder(tf.int32, [None])

        # Index of character (or <RARE> or <STOP> tokens)
        input_ix = tf.placeholder(tf.int32, [None, max_input_length + 1], name = 'input_ix')
        
        # Dense encoding of the characters
        # Note: not <START> token in input, so don't need final row of embedding matrix
        input_dense = tf.nn.embedding_lookup(embedding_matrix[:-1, :], input_ix) 

        # One-hot encoding of the character if it's a 'non-vanilla' char
        input_oh = tf.placeholder(tf.float32, [None, max_input_length + 1, max_nv_chars], name = 'input_oh')

        # Input that will actually be passed to the neural net
        input_x = tf.concat([input_dense, input_oh], axis = 2)

        # Take input sequence and produce the 'final state' that will be used to generate the output tensor.
        # We throw away the 'outputs' from this RNN since it's just collecting information into a state.
        _, final_state = tf.nn.dynamic_rnn(cell            = rnn_encoder_multi_cell,
                                           inputs          = input_x,
                                           sequence_length = input_length,
                                           dtype           = tf.float32)

    # Different net for generating output than for absorbing input
    with tf.variable_scope('decoder') as scope:

        ## Setup tensors for the output sequence
        output_length = tf.placeholder(tf.int32, [None])
        output_y = tf.placeholder(tf.float32, [None, max_output_length + 1, num_output_tokens],
                                  name = 'output_y')  # Correct output sequence, with <STOP> token

        rnn_decoder_cell_1 = tf.contrib.rnn.BasicLSTMCell(layer_width_1)
        rnn_decoder_cell_2 = tf.contrib.rnn.BasicLSTMCell(layer_width_2)
        rnn_decoder_multi_cell = tf.contrib.rnn.MultiRNNCell([rnn_decoder_cell_1, rnn_decoder_cell_2])

        output_layer_w = tf.Variable(tf.truncated_normal([layer_width_2, num_output_tokens], mean = 0.1, stddev = 0.02))
        output_layer_b = tf.Variable(tf.zeros([num_output_tokens]))

        # Input to the decoder
        decoder_input_ix = tf.placeholder(dtype = tf.int32, shape = [None, max_output_length + 1])
        decoder_input_dense = tf.nn.embedding_lookup(embedding_matrix, decoder_input_ix)

        layer_2_outputs, _ = tf.nn.dynamic_rnn(cell   = rnn_decoder_multi_cell,
                                               inputs = decoder_input_dense,
                                               initial_state = final_state,
                                               sequence_length = output_length,
                                               dtype = tf.float32)

        output_y_hat_logits = tf.tensordot(layer_2_outputs, output_layer_w, 1) + output_layer_b

    # Loss measurement
    # We have examples of different sizes in the batch.
    # We want to equally weight examples, rather than equally weighting characters of output.
    flattened_y = tf.reshape(output_y, [-1, num_output_tokens])
    flattened_y_hat_logits = tf.reshape(output_y_hat_logits, [-1, num_output_tokens])
    y_hat_weights = tf.reshape(tf.cast(output_length, dtype=tf.float32) * tf.reduce_max(output_y, axis=2), [-1])
    weight_mask = tf.placeholder(tf.float32, [None, max_output_length+1])
    flattened_weight_mask = tf.reshape(weight_mask, [-1])
    #batch_weights = tf.reciprocal(tf.cast(output_length, dtype = tf.float32))


    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels = flattened_y,
                                                         logits = flattened_y_hat_logits,
                                                         weights = flattened_weight_mask)

    tf.summary.scalar('cross entropy', cross_entropy_loss)
    
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

    # Actually setup tensorflow, and the tensorboard logging
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    # Variables to keep track of training
    batches_trained = 0
    samples_trained = 0


    def make_feed_dict(input_output_strings):

        # Assign memory now
        batch_size = len(input_output_strings)
        input_ix_batch = np.zeros([batch_size, max_input_length+1])
        input_oh_batch = np.zeros([batch_size, max_input_length+1, max_nv_chars])
        output_y_batch = np.zeros([batch_size, max_output_length+1, num_output_tokens])
        decoder_input_ix_batch = np.zeros([batch_size, max_output_length+1])
        input_length_batch  = np.zeros([batch_size])
        output_length_batch = np.zeros([batch_size])
        weight_mask_batch = np.zeros([batch_size, max_output_length+1])


        for (n, (input_string, output_string)) in enumerate(input_output_strings):
            if len(input_string) > max_input_length:
                raise ValueError("input_string length {} is greater than max_input_length {}".format(len(input_string), max_input_length))
            if len(output_string) > max_output_length:
                raise ValueError("output_string length {} is greater than max_output_length {}".format(len(output_string), max_output_length))

            # We need to convert input / output to suitable encodings, and pad to the standard length

            # Encoding of input:
                
            # Flag which characters are non-vanilla and assign them indices
            nv_characters = set([c for c in input_string if not train.is_vanilla(c)])
            if len(nv_characters) > max_nv_chars:
                raise ValueError('Input string contained {} non-vanilla characters. Maximum we are set up for is {}.\nInput: {}'.format(len(nv_characters), max_nv_chars, input_string))
            nv_char_to_ix = {}
            nv_ix_to_char = {}
            for (c, ix) in zip(nv_characters, np.random.choice(max_nv_chars, size=len(nv_characters), replace=False)):
                nv_char_to_ix[c] = ix
                nv_ix_to_char[ix] = c

            input_pad_length = max_input_length - len(input_string) + 1 # +1 for <STOP> token

            # Index of input character - used to index into the embedding matrix (padded with at least one <STOP> token)
            input_ix_concrete = np.array([train.get_index_of_char(c) for c in input_string] + [stop_ix for _ in range(input_pad_length)])

            # Non-vanilla characters also get a one-hot index passed into the net
            input_one_hot_concrete = np.zeros(shape = [len(input_string) + input_pad_length, max_nv_chars])
            for (i, c) in enumerate(input_string):
                if not train.is_vanilla(c):
                    input_one_hot_concrete[i, nv_char_to_ix[c]] = 1.0

            # Encoding of output:

            output_pad_length = max_output_length - len(output_string) + 1  # +1 for <STOP> token
            output_labels_one_hot_concrete = np.zeros(shape = [output_pad_length + len(output_string), train.num_vanilla_chars + max_nv_chars + 1])
            for (i, c) in enumerate(output_string):
                if train.is_vanilla(c):
                    output_labels_one_hot_concrete[i, train.get_index_of_char(c)] = 1.0
                else:
                    output_labels_one_hot_concrete[i, train.num_vanilla_chars + nv_char_to_ix[c]] = 1.0
            # Pad with <STOP> tokens
            for i in range(len(output_string), output_pad_length + len(output_string)):
                output_labels_one_hot_concrete[len(output_string), stop_output_ix] = 1.0

            weight_mask_batch[n, :] = np.array([(1.0/len(output_string)) if i < len(output_string) else 0.0 for i in range(output_pad_length + len(output_string))])

            # Encoding of decoder input:

            decoder_input_pad_length = max_output_length - len(output_string) + 1
            decoder_input_ix_concrete = np.array([start_ix] + [train.get_index_of_char(c) for c in output_string]
                                        + [stop_ix for _ in range(output_pad_length - 1)])

            input_ix_batch[n, :] = input_ix_concrete
            input_oh_batch[n, :, :] = input_one_hot_concrete
            output_y_batch[n, :, :] = output_labels_one_hot_concrete
            decoder_input_ix_batch[n, :] = decoder_input_ix_concrete
            input_length_batch[n]  = len(input_string)  + 1   # +1 for <STOP> token
            output_length_batch[n] = len(output_string) + 1   # +1 for <STOP> token

        return { input_ix : input_ix_batch,
                 input_oh : input_oh_batch,
                 output_y : output_y_batch,
                 decoder_input_ix : decoder_input_ix_batch,
                 input_length  : input_length_batch,
                 output_length : output_length_batch,
                 weight_mask   : weight_mask_batch
               }



    def train_one_batch():
        nonlocal batches_trained, samples_trained
        #input_ix_batch, input_oh_batch, output_y_batch, input_length_batch, output_length_batch = train.next_batch(1)

        input_output_strings = train.next_batch(32)

        feed_dict = make_feed_dict(input_output_strings)

        optimizer.run(feed_dict = feed_dict, session = sess)

        batches_trained += 1
        samples_trained += 1

        if batches_trained % 5 == 0:
            summary = sess.run(merged, feed_dict = feed_dict)
            train_writer.add_summary(summary, batches_trained)

    for i in range(max_batches):
        print("i: {}".format(i))
        train_one_batch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                               'tensorflow/norm_net/logs'),
                        help='Summaries log directory')
    parser.add_argument('--max_batches', type=int,
                        default = 1000,
                        help='Maximum number of batches to train on')
    parser.add_argument('--mini_dataset', dest='mini_dataset', action='store_true')
    flags = parser.parse_args()
    main(flags.log_dir, flags.max_batches, flags.mini_dataset)
