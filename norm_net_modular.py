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

#    def build(self):
#        # Call to superclass builds the recurrent part of the net
#        super(Encoder, self).build_core_rnn()
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
        use_embedding_matrix : bool, whether or not to pass input through an embedding matrix first.

    """

    def __init__(self, params, cell_type = 'LSTM', use_embedding_matrix = True):
        _assert_like_parameters(params)
        _assert_valid_cell_type(cell_type)

        self._max_input_size  = params['max_input_size']
        self._max_output_size = params['max_output_size']

        self.built = False
        self.training_dataset = dataset.TrainingDataset(line_limit = 1000, max_input_size = self._max_input_size,
                                                        max_output_size = self._max_output_size)

    #def build(self):


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
        params = _example_params
        net = Encoder(params, cell_type='LSTM')
        net.build_core_rnn()
        self.assertTrue(net.built)
        
        input_embedded = tf.placeholder(shape=[None, 20, 50], dtype = tf.float32)
        seq_lengths = tf.placeholder(shape = [None], dtype = tf.int32)
        final_state = net.connect(inputs = input_embedded, seq_lengths = seq_lengths)
        # Check that the shape is what we expect
        self.assertEqual([[[x.value for x in hc.shape] for hc in layer] for layer in final_state],
                         [[[None, 10], [None, 10]], [[None, 20], [None, 20]]])

# FIXME - add a test to inspect the trainable weights at the end of this!
# FIXME - add automatic naming of variables?
