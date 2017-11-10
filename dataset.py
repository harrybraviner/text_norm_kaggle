import numpy as np
import re, unittest

def is_vanilla_char(c):
    if ord(c) >= ord('a') and ord(c) <= ord('z'):
        return True
    if ord(c) >= ord('A') and ord(c) <= ord('Z'):
        return True
    if ord(c) >= ord('0') and ord(c) <= ord('9'):
        return True
    if ord(c) in [ord(x) for x in [' ', '\'']]: return True

    return False

splitter_re = re.compile(r'(\d+),(\d+),"(.*)","(.*)","(.*)"')

def split_line(l):
    return splitter_re.match(l).groups()

class TrainingDataset:

    def __init__(self, location = 'data/en_train.csv', train_fraction = 0.7, line_limit = None,
                 max_input_size = None, max_output_size = None):

        self._max_output_size = max_output_size
        self._max_input_size  = max_input_size

        with open(location, 'rt') as f:
            f.readline()    # Throw away the header line
            if line_limit is None:
                self._all_data = [split_line(l)[3:5] for l in f]
            else:
                # Used if we want to run with only a small amount of data to check the code works
                lines = [next(f) for _ in range(line_limit)]
                self._all_data = [split_line(l)[3:5] for l in lines]

        # Filter to input and output sizes
        self._all_data = [(i, o) for (i, o) in self._all_data if     (max_input_size  is None or len(i) <= max_input_size )
                                                                 and (max_output_size is None or len(o) <= max_output_size) ]

        self._N_total = len(self._all_data)
        self._N_train = int(train_fraction * self._N_total)
        self._N_val = self._N_total - self._N_train

        self._training_cursor = 0   # Used to keep track of which example to serve next

        # How many of each character does the input contain?
        self._char_counts = {}
        for l in self._all_data[:self._N_train]:
            cs = set(l[0])
            for c in cs:
                if c not in self._char_counts:
                    self._char_counts[c] = 0
                self._char_counts[c] += 1

        # Assign numeric indices to each character
        self._c_to_ix = {}
        self._ix_to_c = {}
        lc_chars = [chr(i) for i in range(ord('a'), ord('z')+1)]
        uc_chars = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        num_chars = [chr(i) for i in range(ord('0'), ord('9')+1)]
        misc_chars = [' ', '\'']
        vanilla_chars = lc_chars + uc_chars + num_chars + misc_chars
        self._vanilla_chars = set(vanilla_chars)    # Used for fast testing of whether a char is vanilla
        self._num_vanilla_chars = len(vanilla_chars)

        non_vanilla_chars = [c for c in self._char_counts if c not in vanilla_chars and self._char_counts[c] >= 10]
        non_vanilla_chars.sort()

        ix_cursor = 0
        for c in vanilla_chars + non_vanilla_chars:
            self._c_to_ix[c] = ix_cursor
            self._ix_to_c[ix_cursor] = c
            ix_cursor += 1
        self._rare_ix = ix_cursor
        self._stop_ix = self._rare_ix + 1

        # For characters that appear rarely in the training data, just assign them to the <RARE> token
        for c in self._char_counts:
            if self._char_counts[c] < 10:
                self._c_to_ix[c] = self._rare_ix

        self._max_nv_chars = 10 # Used to set the width of the one-hot coding for nv-chars
                                # This is the maximum number of distinct non-vanilla chars we can support in a single input

    @property
    def num_input_chars(self):
        # Number of vanilla characters + non-vanilla input characters + 2 (<RARE> and <STOP>)
        return self._stop_ix + 1
    
    @property
    def max_nv_chars(self):
        return self._max_nv_chars

    @property
    def rare_token_ix(self):
        return self._rare_ix

    @property
    def stop_input_token_ix(self):
        return self._stop_ix

    @property
    def num_output_chars(self):
        # The +1 is for the <STOP> token
        return self._num_vanilla_chars + self.max_nv_chars + 1

    @property
    def stop_output_token_ix(self):
        return self._num_vanilla_chars + self._max_nv_chars

    @property
    def start_output_token_ix(self):
        return self.stop_output_token_ix + 1

    def convert_input_output_pair(self, input_string, output_string, input_pad_length = None, output_pad_length = None):

        if input_pad_length is not None and len(input_string) > input_pad_length:
            raise ValueError("input_string length {} is greater than input_pad_length {}".format(len(input_string), input_pad_length))

        if output_pad_length is not None and len(output_string) > output_pad_length:
            raise ValueError("output_string length {} is greater than output_pad_length {}".format(len(output_string), output_pad_length))

        # Flag which characters are non-vanilla and assign them indices
        nv_characters = set([c for c in input_string if c not in self._vanilla_chars])
        if len(nv_characters) > self._max_nv_chars:
            raise ValueError('Input string contained {} non-vanilla characters. Maximum set up for is {}.\nInput: {}'.format(len(nv_characters), self._max_nv_chars, input_string))
        nv_char_to_ix = {}
        nv_ix_to_char = {}
        for (c, ix) in zip(nv_characters, np.random.choice(self._max_nv_chars, size=len(nv_characters), replace=False)):
            nv_char_to_ix[c] = ix
            nv_ix_to_char[ix] = c

        if input_pad_length is None:
            pad_size = 1 # For the <STOP> token
        else:
            pad_size = input_pad_length - len(input_string) + 1

        # Array to index into embedding array
        input_ix = np.array([self._c_to_ix[c] for c in input_string])
        input_ix = np.concatenate([input_ix, [self._stop_ix]*pad_size])  # Pad with <STOP> tokens (always gets at least one)

        # One-hot encoding
        input_one_hot = np.zeros(shape=[len(input_string) + pad_size, self._max_nv_chars]) # +1 because of the <STOP> token
        for (i, c) in enumerate(input_string):
            if c in nv_characters:
                input_one_hot[i, nv_char_to_ix[c]] = 1.0

        # Label
        if output_pad_length is None:
            pad_size = 1 # For the <STOP> token
        else:
            pad_size = output_pad_length - len(output_string) + 1
        label_one_hot = np.zeros(shape=[len(output_string)+pad_size, self.num_output_chars])
        for (i, c) in enumerate(output_string):
            if c in self._vanilla_chars:
                label_one_hot[i, self._c_to_ix[c]] = 1.0
            else:
                label_one_hot[i, len(self._vanilla_chars) + nv_char_to_ix[c]] = 1.0
        label_one_hot[-1, self.stop_output_token_ix] = 1

        return input_ix, input_one_hot, label_one_hot, len(input_string), len(output_string)

    def next_training_example(self):

        input_string, output_string = self._all_data[self._training_cursor]

        self._training_cursor = (self._training_cursor + 1) % self._N_train

        return self.convert_input_output_pair(input_string, output_string)

    def next_batch(self, batch_size):

        # FIXME - need to also pass the actual lengths of the examples

        input_output_data = [self.next_training_example() for _ in range(batch_size)]

        input_ix = np.stack([x[0] for x in input_output_data])
        input_oh = np.stack([x[1] for x in input_output_data])
        output_y = np.stack([x[2] for x in input_output_data])
        input_length  = np.stack([x[3] for x in input_output_data])
        output_length = np.stack([x[4] for x in input_output_data])

        return (input_ix, input_oh, output_y, input_length, output_length)



class TestsForTrainingDataset(unittest.TestCase):

    def test_construction_and_draw_example(self):
        train = TrainingDataset(line_limit = 1000)

        _, _, _, _, _ = train.next_training_example()
        _, _, _, _, _ = train.next_training_example()
        _, _, _, _, _ = train.next_training_example()
        _, _, _, _, _ = train.next_training_example()
