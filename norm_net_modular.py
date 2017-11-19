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
    'embedding_size'  : 32,
    'max_nv_chars'    : 10,
    'cell_type'    : 'LSTM',
    'layer_size_1' : 10,
    'layer_size_2' : 20
}

def _assert_valid_cell_type(x):
    if x not in ['Basic', 'LSTM', 'GRU']:
        raise ValueError('Allowed cell types are "Basic", "LSTM" and "GRU". Got {}.'.format(x))

def _assert_like_parameters(x):
    if type(x) != dict:
        raise ValueError('Parameters should be a dict.')
    required_keys = ['num_layers', 'n_out', 'max_input_size', 'max_nv_chars', 'cell_type', 'max_output_size']
    actual_keys = x.keys()
    for k in required_keys:
        if k not in actual_keys:
            raise ValueError('Key {} is missing from parameters.'.format(k))
    _assert_valid_cell_type(x['cell_type'])
    for i in range(1, x['num_layers'] + 1):
        if 'layer_size_{}'.format(i) not in actual_keys:
            raise ValueError('Key {} is missing from parameters.'.format('layer_size_{}'.format(i)))

def _get_shape(t):
    """Get the shape of a tensor as a list,
    since that's more convenient for our tests.
    """
    return [x.value for x in t.shape]

def _assert_like_multi_cell_state(x, layer_sizes, cell_type):
    """Check that the state passed into a function is of the expected shape.
    Note: I am assuming that x comes from a multi_cell. If not, then it will be
    a tensor, rather than an iterable of tensors, and this function will fail.
    Don't blindly reuse this function if you need to cover that use case!
    
    Args:
        x: Tuple or list of tensors
        layer_sizes: The layer sizes we want to check for compatability with.
        cell_type: Different cell types produce different shapes of output
    """
    if cell_type == 'Basic' or cell_type == 'GRU':  # Basic and GRU states have same shape
        try:
            shapes = [_get_shape(layer) for layer in x]
        except:
            raise ValueError('State did not have expected form for Basic or GRU rnn state. Got:\n{}'.format(x))
        batch_size = shapes[0][0]
        for (i, s) in enumerate(shapes):
            if s[0] != batch_size:
                raise ValueError('Inconsistent batch sizes. Expected {} based on 0th layer, but found {} in layer {}.'
                                 .format(batch_size, s[0], i))
            if s[1] != layer_sizes[i]:
                raise ValueError('State size at layer {} was {}, but layer size is {}.'.format(i, s[1], layer_sizes[i]))
        return
    elif cell_type == 'LSTM':
        try:
            shapes = [[_get_shape(xx) for xx in layer] for layer in x]
        except:
            raise ValueError('State did not have expected form for LSTM state. Got:\n{}'.format(x))
        batch_size = shapes[0][0][0]
        for (i, s) in enumerate(shapes):
            if s[0][0] != batch_size:
                raise ValueError('Inconsistent batch sizes. Expected {} based on 0th layer, but found {} in c in layer {}.'
                                 .format(batch_size, s[0][0], i))
            if s[1][0] != batch_size:
                raise ValueError('Inconsistent batch sizes. Expected {} based on 0th layer, but found {} in h in layer {}.'
                                 .format(batch_size, s[1][0], i))
            if s[0][1] != layer_sizes[i]:
                raise ValueError('State size in c at layer {} was {}, but layer size is {}.'.format(i, s[0][1], layer_sizes[i]))
            if s[1][1] != layer_sizes[i]:
                raise ValueError('State size in h at layer {} was {}, but layer size is {}.'.format(i, s[1][1], layer_sizes[i]))
        return
    else:
        raise ValueError('Allowed cell types are "Basic", "LSTM" and "GRU". Got {}.'.format(x))

class RecurrentNet:
    """Base class - do not instantiate this.
    Contains the common code for defining the RNN part of the Encoder and Decoder nets.
    """

    def __init__(self, params):
        _assert_like_parameters(params)
        self._cell_type = params['cell_type']
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

    def __init__(self, params):
        super(Encoder, self).__init__(params)

    def build(self):
        with tf.variable_scope('encoder'):
            # Call to superclass builds the recurrent part of the net
            super(Encoder, self).build_core_rnn()

    def connect(self, inputs, seq_lengths):
        """Returns the output_state tensor that the encoder produces.

        Args:
            inputs: Rank 3 tensor of dimensions [batch_size, max seq length, embedding size + max_nv_chars]
            seq_lengths: Rank 1 tensor of dimensions [batch_size] giving the actual sequence lengths
        """
        if len(inputs.shape) != 3:
            raise ValueError('inputs must be a rank 3 tensor, but rank is {}'.format(len(inputs.shape)))
        if len(seq_lengths.shape) != 1:
            raise ValueError('seq_lengths must be a rank 1 tensor, but rank is {}'.format(len(seq_lengths.shape)))

        if not self.built:
            # Construct the rnn cells
            self.build()

        with tf.variable_scope('encoder'):
            _, output_state = tf.nn.dynamic_rnn(cell            = self._multi_cell,
                                                inputs          = inputs,
                                                sequence_length = seq_lengths,
                                                dtype           = tf.float32)
        return output_state

class Decoder(RecurrentNet):

    def __init__(self, params):
        super(Decoder, self).__init__(params)

    def build(self):
        with tf.variable_scope('decoder'):
            super(Decoder, self).build_core_rnn()

            # Fully connected net for output
            # Xavier initialization of weights
            n_in = self._layer_sizes[-1]
            n_out = self._n_out
            c = np.sqrt(6.0 / (n_in + n_out))
            self._fc_weights = tf.Variable(tf.random_uniform(shape = [n_in, n_out], minval = -c, maxval = +c, dtype=np.float32))
            self._fc_bias = tf.Variable(tf.zeros(shape = [n_out], dtype=np.float32))

        self.built = True

    def connect(self, initial_state, hint_inputs, seq_lengths):
        """Returns the tensor of output sequences that the decoder produces.

        Args:
            initial_state: The final state of the encoder network. Shape depends on the number of layers and type of cells.
            hint_inputs: The 'previous' characters. i.e. these are the target output sequences, delayed by one time unit and with a <START> token prepended.
            seq_lengths: Rank 1 tensor of dimensions [batch_size] giving the actual sequence lengths
        """
        _assert_like_multi_cell_state(initial_state, self._layer_sizes, self._cell_type)
        if len(seq_lengths.shape) != 1:
            raise ValueError('seq_lengths must be a rank 1 tensor, but rank is {}'.format(len(seq_lengths.shape)))

        if not self.built:
            # Construct the rnn cells and the final (fully connected) layer weights
            self.build()

        with tf.variable_scope('decoder'):
            self._rnn_output, _ = tf.nn.dynamic_rnn(cell      = self._multi_cell,
                                              inputs          = hint_inputs,
                                              sequence_length = seq_lengths,
                                              initial_state   = initial_state,
                                              dtype           = tf.float32)
        output_y_logits = tf.tensordot(self._rnn_output, self._fc_weights, axes = [[2], [0]]) + self._fc_bias

        return output_y_logits

class TranslationNet:
    """This constructs both an Encoder and Decoder, and handles passing
    inputs to them via a common embedding matrix.
    
    Args:
        embedding_size : int, the size of the embedding vectors to use for the characters. If None, the embedding matrix is the identity matrix (i.e. one-hot encodings are passed straight to the RNN).

    """

    def __init__(self, params, mini_dataset = False):
        _assert_like_parameters(params)

        self._params = params   # Save for passeing to encoder and decoder
        self._max_input_size  = params['max_input_size']
        self._padded_max_input_size = self._max_input_size + 1  # Padding with <STOP> token
        self._max_output_size = params['max_output_size']
        self._padded_max_output_size = self._max_output_size + 1  # Padding with <STOP> token
        self._cell_type = params['cell_type']
        self._embedding_size = params['embedding_size']
        self._max_nv_chars = params['max_nv_chars']

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
        # For the decoder output, non-vanilla characters must be coded using the one-hot token pased in as input
        self._output_stop_ix = self.training_dataset.num_vanilla_chars + self._max_nv_chars
        self._num_output_tokens = self.training_dataset.num_vanilla_chars + self._max_nv_chars + 1

    def build(self):
        self._unnorm_ix = tf.placeholder(shape = [None, self._padded_max_input_size], dtype = tf.int32)
        self._unnorm_nv = tf.placeholder(shape = [None, self._padded_max_input_size, self._max_nv_chars], dtype = tf.float32)
        self._norm_hint_ix = tf.placeholder(shape = [None, self._padded_max_output_size], dtype = tf.int32)
        if self._embedding_size is not None:
            self._embedding_matrix = tf.Variable(tf.truncated_normal([self._num_input_tokens, self._embedding_size], mean=0.1, stddev=0.02))
            self._encoder_input = tf.concat([tf.nn.embedding_lookup(self._embedding_matrix, self._unnorm_ix),
                                             self._unnorm_nv], axis = 2)
            self._decoder_hint_input = tf.nn.embedding_lookup(self._embedding_matrix, self._norm_hint_ix)
        else:
            self._encoder_input = tf.concat([tf.one_hot(indices = self._unnorm_ix, depth = self._num_input_tokens),
                                             self._unnorm_nv], axis = 2)
            self._decoder_hint_input = tf.one_hot(indices = self._norm_hint_ix, depth = self._num_input_tokens)

        self._unnorm_seq_lengths = tf.placeholder(shape = [None], dtype=tf.int32)
        self._norm_seq_lengths = tf.placeholder(shape = [None], dtype=tf.int32)

        self._encoder = Encoder(self._params)
        self._decoder = Decoder(self._params)

        # 'Wire up' the circuit (self._unnorm_ix, self._unnorm_seq_lengths) --> self._encoder_state_out
        self._encoder_state_out = self._encoder.connect(self._encoder_input, seq_lengths = self._unnorm_seq_lengths)
        self._decoder_logits_out = self._decoder.connect(self._encoder_state_out, self._decoder_hint_input,
                                                         seq_lengths = self._norm_seq_lengths)

        # FIXME - need to do the decoder in 'unhinted' mode

        self.built = True

    def convert_unnorm_to_input_ix(self, unnorm_string):
        """Takes a string and converts it to the input format expected by the encoder.
        This is a one-hot vector, plus an 'at-most-one-hot' vector for encoding
        non-vanilla characters.

        Args:
            unnorm_string: The string we wish to convert.
        """
        if len(unnorm_string) > self._max_input_size:
            raise ValueError('Input string had length {}, but max allowed is {}.'.format(len(unnorm_string), self._max_input_size))
        output = np.zeros(dtype = np.int32, shape = [self._padded_max_input_size])
        for (i, c) in enumerate(unnorm_string):
            output[i] = self.training_dataset.get_index_of_char(c)
        output[len(unnorm_string)] = self._input_stop_ix
        return output

    def convert_unnorm_to_input_nv(self, nv_to_ix, unnorm_string):
        """Takes a string and converts the non-vanilla characters in it to the relevant one-hot
        tokens for input into the encoder. Vanilla characters just produce a zero vector.

        Args:
            nv_to_ix: Dictionary mapping non-vanilla characters in the string to one-hot indices.
            unnorm_string: The string we wish to convert.
        """
        if len(unnorm_string) > self._max_input_size:
            raise ValueError('Input string had length {}, but max allowed is {}.'.format(len(unnorm_string), self._max_input_size))
        output = np.zeros(dtype = np.float32, shape = [self._padded_max_input_size, self._max_nv_chars])
        for (i, c) in enumerate(unnorm_string):
            if not self.training_dataset.is_vanilla(c):
                output[i, nv_to_ix[c]] = 1.0
        return output

    def convert_norm_to_input(self, norm_string):
        """Takes a normalised string and converts it to a sequence of tokens indices for
        input to the decoder. The <START> token is placed at the beginning and the whole
        sequence is 'delayed' by one.

        Args:
            norm_string: The string we wish to convert.
        """
        if len(norm_string) > self._max_output_size:
            raise ValueError('Input string had length {}, but max allowed is {}.'.format(len(norm_string), self._max_output_size))
        output = np.zeros(dtype = np.int32, shape = [self._padded_max_output_size])
        output[0] = self._input_start_ix
        for (i, c) in enumerate(norm_string):
            output[i+1] = self.training_dataset.get_index_of_char(c)
        return output

    def convert_norm_to_output(self, nv_to_ix, norm_string):
        """Takes a normalised string and converts it to the one-hot format that we are
        training the decoder to produce.

        Args:
            nv_to_ix: Dictionary mapping non-vanilla characters in the string to one-hot indices.
            norm_string: The string we wish to convert.
        """
        if len(norm_string) > self._max_output_size:
            raise ValueError('Input string had length {}, but max allowed is {}.'.format(len(norm_string), self._max_output_size))
        output = np.zeros(dtype = np.float32, shape = [self._padded_max_output_size, self._num_output_tokens])
        for (i, c) in enumerate(norm_string):
            if self.training_dataset.is_vanilla(c):
                ix = self.training_dataset.get_index_of_char(c)
            else:
                try:
                    # Non-vanilla output tokens come straight after vanilla character tokens
                    ix = self.training_dataset.num_vanilla_chars + nv_to_ix[c]
                except:
                    raise ValueError('Input string contained character {}, which was neither vanilla, nor present in the nv_to_ix dictionary.'.format(c))
            output[i, ix] = 1.0
        # Append <STOP> token at the end
        output[len(norm_string), self._output_stop_ix] = 1.0
        return output

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
            'embedding_size'  : 32,
            'max_nv_chars'    : 10,
            'cell_type'    : 'LSTM',
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
            'embedding_size'  : 32,
            'max_nv_chars'    : 10,
            'cell_type'    : 'LSTM',
            'n_out' : 50,
            'layer_size_1' : 10,
        }
        with self.assertRaises(ValueError):
            _assert_like_parameters(params)

class AssertLikeMultiCellStateTests(unittest.TestCase):

    def test_basic_rnn_correct(self):
        with tf.variable_scope('shape_basic_test_correct'):
            c1 = tf.contrib.rnn.BasicRNNCell(10)
            c2 = tf.contrib.rnn.BasicRNNCell(13)
            c3 = tf.contrib.rnn.BasicRNNCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            _assert_like_multi_cell_state(out_state, layer_sizes = [10, 13, 20], cell_type='Basic')

    def test_basic_rnn_fail_1(self):
        with tf.variable_scope('shape_basic_test_fail_1'):
            c1 = tf.contrib.rnn.BasicRNNCell(10)
            c2 = tf.contrib.rnn.BasicRNNCell(13)
            c3 = tf.contrib.rnn.BasicRNNCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            with self.assertRaises(ValueError):
                _assert_like_multi_cell_state(out_state, layer_sizes = [10, 12, 20], cell_type='Basic')

    def test_lstm_correct(self):
        with tf.variable_scope('shape_lstm_test_correct'):
            c1 = tf.contrib.rnn.BasicLSTMCell(10)
            c2 = tf.contrib.rnn.BasicLSTMCell(13)
            c3 = tf.contrib.rnn.BasicLSTMCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            _assert_like_multi_cell_state(out_state, layer_sizes = [10, 13, 20], cell_type='LSTM')

    def test_lstm_fail_1(self):
        with tf.variable_scope('shape_lstm_test_fail_1'):
            c1 = tf.contrib.rnn.BasicLSTMCell(10)
            c2 = tf.contrib.rnn.BasicLSTMCell(13)
            c3 = tf.contrib.rnn.BasicLSTMCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            with self.assertRaises(ValueError):
                _assert_like_multi_cell_state(out_state, layer_sizes = [10, 12, 20], cell_type='LSTM')

    def test_lstm_fail_2(self):
        with tf.variable_scope('shape_lstm_test_fail_2'):
            c1 = tf.contrib.rnn.BasicLSTMCell(10)
            c2 = tf.contrib.rnn.BasicLSTMCell(13)
            c3 = tf.contrib.rnn.BasicLSTMCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            with self.assertRaises(ValueError):
                _assert_like_multi_cell_state(out_state, layer_sizes = [10, 13, 20], cell_type='Basic')

    def test_gru_rnn_correct(self):
        with tf.variable_scope('shape_gru_test_correct'):
            c1 = tf.contrib.rnn.GRUCell(10)
            c2 = tf.contrib.rnn.GRUCell(13)
            c3 = tf.contrib.rnn.GRUCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            _assert_like_multi_cell_state(out_state, layer_sizes = [10, 13, 20], cell_type='GRU')

    def test_gru_rnn_fail_1(self):
        with tf.variable_scope('shape_gru_test_fail_1'):
            c1 = tf.contrib.rnn.GRUCell(10)
            c2 = tf.contrib.rnn.GRUCell(13)
            c3 = tf.contrib.rnn.GRUCell(20)
            mc = tf.contrib.rnn.MultiRNNCell([c1, c2, c3])
            i = tf.placeholder(dtype=tf.float32, shape=[None, 5, 50])

            _, out_state = tf.nn.dynamic_rnn(cell = mc, inputs=i, dtype=tf.float32)
            with self.assertRaises(ValueError):
                _assert_like_multi_cell_state(out_state, layer_sizes = [10, 12, 20], cell_type='GRU')

class RecurrentNetTests(unittest.TestCase):
    
    def test_contruction_and_building(self):
        params = _example_params
        net = RecurrentNet(params)
        net.build_core_rnn()
        self.assertTrue(net.built)

class EncoderTests(unittest.TestCase):
    
    def test_contruction_and_building(self):
        with tf.variable_scope('EncoderTests'):
            params = _example_params
            net = Encoder(params)
            net.build_core_rnn()
            self.assertTrue(net.built)
            
            input_embedded = tf.placeholder(shape=[None, 20, 50], dtype = tf.float32)
            seq_lengths = tf.placeholder(shape = [None], dtype = tf.int32)
            final_state = net.connect(inputs = input_embedded, seq_lengths = seq_lengths)
            # Check that the shape is what we expect
            self.assertEqual([[_get_shape(hc) for hc in layer] for layer in final_state],
                             [[[None, 10], [None, 10]], [[None, 20], [None, 20]]])

class DecoderTests(unittest.TestCase):

    def test_construction_and_building(self):

        with tf.variable_scope('DecoderTests'):
            params = _example_params
            net = Decoder(params)
            net.build()

    def test_encoder_decoder_wiring(self):
        with tf.variable_scope('DecoderEncoderTests'):
            params = _example_params
            encoder = Encoder(params)
            decoder = Decoder(params)

            input_embedded = tf.placeholder(shape=[None, 20, 50], dtype = tf.float32)
            hint_input_embedded = tf.placeholder(shape=[None, 30, 50], dtype = tf.float32)
            seq_lengths = tf.placeholder(shape = [None], dtype = tf.int32)
            final_state = encoder.connect(inputs = input_embedded, seq_lengths = seq_lengths)
            output_y_logits = decoder.connect(final_state, hint_input_embedded, seq_lengths = seq_lengths)

class TranslationNetTests(unittest.TestCase):

    def test_construction_and_geometry(self):
        with tf.variable_scope('TranslationNetTests'):
            params = _example_params
            net = TranslationNet(params, mini_dataset = True)
            net.build()
            self.assertTrue(net.built)

            # Should have 3 special tokens: <RARE>, <START>, and <STOP>
            self.assertEqual(net._num_input_tokens, net.training_dataset.num_non_rare_chars + 3)
            self.assertEqual(_get_shape(net._encoder_input),
                             [None, net._padded_max_input_size, net._embedding_size + net._max_nv_chars])
            # FIXME - it should not be params deciding how many output tokens there are!
            self.assertEqual(_get_shape(net._decoder_logits_out), [None, net._padded_max_output_size, params['n_out']])

    def test_formatting_for_encoder(self):
        net = TranslationNet(_example_params, mini_dataset = True)
        
        unnorm_string = "Hello world\"£"
        unnorm_ix_expected = np.array([33, 4, 11, 11, 14, 62, 22, 14, 17, 11, 3,
                                       net.training_dataset.get_index_of_char('\"'), net._input_rare_ix, net._input_stop_ix]
                                      + [0 for i in range(net._padded_max_input_size - 14)])
        self.assertTrue((net.convert_unnorm_to_input_ix(unnorm_string) == unnorm_ix_expected).all())
        unnorm_oh_expected = np.zeros(dtype=np.float32, shape = [net._padded_max_input_size, _example_params['max_nv_chars']])
        nv_to_ix_dict = {'\"': 2, '£' : 5}
        unnorm_oh_expected[11, 2] = 1.0; unnorm_oh_expected[12, 5] = 1.0
        self.assertTrue((net.convert_unnorm_to_input_nv(nv_to_ix_dict, unnorm_string) == unnorm_oh_expected).all())

    def test_formatting_for_decoder(self):
        net = TranslationNet(_example_params, mini_dataset = True)

        norm_string = "Hello world\"£"
        norm_hint_expected = np.array([net._input_start_ix, 33, 4, 11, 11, 14, 62, 22, 14, 17, 11, 3,
                                      net.training_dataset.get_index_of_char('\"'), net._input_rare_ix]
                                     + [0 for i in range(net._padded_max_output_size - 14)])
        self.assertTrue((net.convert_norm_to_input(norm_string) == norm_hint_expected).all())
        norm_output_expected = np.zeros(dtype = np.float32, shape=[net._padded_max_output_size, net._num_output_tokens])
        norm_output_expected[[i for i in range(14)], [33, 4, 11, 11, 14, 62, 22, 14, 17, 11, 3,
                                                    net.training_dataset.num_vanilla_chars + 2,
                                                    net.training_dataset.num_vanilla_chars + 5,
                                                    net._output_stop_ix]] = 1.0
        nv_to_ix_dict = {'\"': 2, '£' : 5}
        self.assertTrue((net.convert_norm_to_output(nv_to_ix_dict, norm_string) == norm_output_expected).all())

# FIXME - add a test to inspect the trainable weights at the end of this!
