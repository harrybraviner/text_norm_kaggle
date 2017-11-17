#! /usr/bin/python3

import numpy as np
import tensorflow as tf
import dataset
import argparse, os, unittest

_example_params = {
    'num_layers' : 2,
    'n_out' : 50,
    'max_input_size'  : 20,
    'max_output_size' : 30,
    'layer_size_1' : 10,
    'layer_size_2' : 20
}

def _assert_like_parameters(x):
    if type(x) != dict:
        raise ValueError('Parameters should be a dict.')
    required_keys = ['num_layers', 'n_out', 'max_input_size', 'max_output_size']
    actual_keys = x.keys()
    for k in required_keys:
        if k not in actual_keys:
            raise ValueError('Key {} is missing from parameters.'.format(k))
    for i in range(1, x['num_layers'] + 1):
        if 'layer_size_{}'.format(i) not in actual_keys:
            raise ValueError('Key {} is missing from parameters.'.format('layer_size_{}'.format(i)))

def _assert_valid_cell_type(x):
    if x not in ['Basic', 'LSTM', 'GRU']:
        raise ValueError('Allowed cell types are "Basic", "LSTM" and "GRU". Got {}.'.format(x))

class RecurrentNet:
    """Base class - do not instantiate this.
    Contains the common code for defining the RNN part of the Encoder and Decoder nets.
    """

    def __init__(self, params, cell_type = 'LSTM'):
        _assert_like_parameters(params)
        _assert_valid_cell_type(cell_type)
        self._cell_type = cell_type
        self._layer_sizes = [params['layer_size_{}'.format(i+1)] for i in range(params['num_layers'])]
        self._n_out = params['n_out']

        self.built = False

    def _make_rnn_cell(self, size):
        if self._cell_type == 'Basic':
            return tf.contrib.rnn.BasicRNNCell(size)
        elif self._cell_type == 'LSTM':
            return tf.contrib.rnn.BasicLSTMCell(size)
        elif self._cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(size)
        else:
            raise ValueError('Unknown cell type {}'.format(self._cell_type))

    def build_core_rnn(self):
        self._rnn_cells = []
        for size in self._layer_sizes:
            self._rnn_cells.append(self._make_rnn_cell(size))
        self._multi_cell = tf.contrib.rnn.MultiRNNCell(self._rnn_cells)
        self.built = True

class Encoder(RecurrentNet):

    def __init__(self, params, cell_type = 'LSTM'):
        super(Encoder, self).__init__(params, cell_type)

    def build(self):
        # Call to superclass builds the recurrent part of the net
        super(Encoder, self).build_core_rnn()

#        # FIXME - move the stuff below to the Decoder
#        #n_in = self._layer_sizes[-1]
#        #n_out = self._n_out
#        ## Xavier initialization of weights
#        #c = np.sqrt(6.0 / (n_in + n_out))
#        #self._fc_weights = tf.Variable(tf.random_uniform(shape = [n_in, n_out], minval = -c, maxval = +c, dtype=np.float32))
#        #self._fc_bias = tf.Variable(tf.zeros(shape = [n_out], dtype=np.float32))
#        #self._output_y_logits = self._multi_cell
#        self.built = True

    def connect(self, inputs, seq_lengths):
        """Returns the output_state tensor that the encoder produces.

        Args:
            inputs: Rank 3 tensor of dimensions [batch_size, max seq length, embedding size]
            sequence_length: Rank 1 tensor of dimensions [batch_size] giving the actual sequence lengths
        """
        if len(inputs.shape) != 3:
            raise ValueError('inputs must be a rank 3 tensor, but rank is {}'.format(len(inputs.shape)))
        if len(seq_lengths.shape) != 1:
            raise ValueError('seq_lengths must be a rank 1 tensor, but rank is {}'.format(len(seq_lengths.shape)))

        if not self.built:
            # Construct the rnn cells
            self.build()

        _, output_state = tf.nn.dynamic_rnn(cell            = self._multi_cell,
                                            inputs          = inputs,
                                            sequence_length = seq_lengths,
                                            dtype           = tf.float32)
        return output_state

class TranslationNet:
    """This constructs both an Encoder and Decoder, and handles passing
    inputs to them via a common embedding matrix.
    
    Args:
        embedding_size : int, the size of the embedding vectors to use for the characters. If None, the embedding matrix is the identity matrix (i.e. one-hot encodings are passed straight to the RNN).

    """

    def __init__(self, params, cell_type = 'LSTM', embedding_size = 32, mini_dataset = False):
        _assert_like_parameters(params)
        _assert_valid_cell_type(cell_type)

        self._params = params   # Save for passeing to encoder and decoder
        self._max_input_size  = params['max_input_size']
        self._max_output_size = params['max_output_size']
        self._cell_type = cell_type
        self._embedding_size = embedding_size

        self.built = False

        # Load the training dataset - will need some properties of this for making the embedding matrix
        line_limit = 1000 if mini_dataset else None
        self.training_dataset = dataset.TrainingDataset(line_limit = line_limit, max_input_size = self._max_input_size,
                                                        max_output_size = self._max_output_size)

        # The unnormalized input to the encoder (and the hint input to the decoder)
        # can be a non-rare character, or the tokens <RARE>, <STOP>, or <START>
        self._input_rare_ix  = self.training_dataset.num_non_rare_chars
        self._input_stop_ix  = self.training_dataset.num_non_rare_chars + 1
        self._input_start_ix = self.training_dataset.num_non_rare_chars + 2
        self._num_input_tokens = self._input_start_ix + 1

    def build(self):
        self._unnorm_ix = tf.placeholder(shape = [None, self._num_input_tokens], dtype = tf.int32)
        if self._embedding_size is not None:
            self._embedding_matrix = tf.Variable(tf.truncated_normal([self._num_input_tokens, self._embedding_size], mean=0.1, stddev=0.02))
            self._encoder_input = tf.nn.embedding_lookup(self._embedding_matrix, self._unnorm_ix)
        else:
            self._encoder_input = tf.one_hot(indices = self._unnorm_ix, depth = self._num_input_tokens)

        self._unnorm_seq_lengths = tf.placeholder(shape = [None], dtype=tf.int32)

        self._encoder = Encoder(self._params, cell_type = self._cell_type)

        # 'Wire up' the circuit (self._unnorm_ix, self._unnorm_seq_lengths) --> self._encoder_state_out
        self._encoder_state_out = self._encoder.connect(self._encoder_input, seq_lengths = self._unnorm_seq_lengths)

        # FIXME - need to do the decoder

        self.built = True


def _get_shape(t):
    """Get the shape of a tensor as a list,
    since that's more convenient for our tests.
    """
    return [x.value for x in t.shape]

class LikeParameterTests(unittest.TestCase):

    def test_success(self):
        params = _example_params
        try:
            _assert_like_parameters(params)
        except:
            self.fail('Valid param dict through an exception!')

    def test_with_missing_field(self):
        params = {
            'num_layers' : 2,
            'max_input_size'  : 20,
            'max_output_size' : 30,
            'layer_size_1' : 10,
            'layer_size_2' : 20
        }
        with self.assertRaises(ValueError):
            _assert_like_parameters(params)

    def test_with_missing_layer_size_field(self):
        params = {
            'num_layers' : 2,
            'max_input_size'  : 20,
            'max_output_size' : 30,
            'n_out' : 50,
            'layer_size_1' : 10,
        }
        with self.assertRaises(ValueError):
            _assert_like_parameters(params)

class RecurrentNetTests(unittest.TestCase):
    
    def test_contruction_and_building(self):
        params = _example_params
        net = RecurrentNet(params, cell_type='LSTM')
        net.build_core_rnn()
        self.assertTrue(net.built)

class EncoderTests(unittest.TestCase):
    
    def test_contruction_and_building(self):
        with tf.variable_scope('EncoderTests'):
            params = _example_params
            net = Encoder(params, cell_type='LSTM')
            net.build_core_rnn()
            self.assertTrue(net.built)
            
            input_embedded = tf.placeholder(shape=[None, 20, 50], dtype = tf.float32)
            seq_lengths = tf.placeholder(shape = [None], dtype = tf.int32)
            final_state = net.connect(inputs = input_embedded, seq_lengths = seq_lengths)
            # Check that the shape is what we expect
            self.assertEqual([[_get_shape(hc) for hc in layer] for layer in final_state],
                             [[[None, 10], [None, 10]], [[None, 20], [None, 20]]])

class TranslationNetTests(unittest.TestCase):

    def test_construction_and_geometry(self):
        with tf.variable_scope('TranslationNetTests'):
            params = _example_params
            net = TranslationNet(params, mini_dataset = True)
            net.build()
            self.assertTrue(net.built)

            # Should have 3 special tokens: <RARE>, <START>, and <STOP>
            self.assertEqual(_get_shape(net._unnorm_ix), [None, net.training_dataset.num_non_rare_chars + 3])

# FIXME - add a test to inspect the trainable weights at the end of this!
# FIXME - add automatic naming of variables?
