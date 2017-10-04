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

    def __init__(self, location = 'data/en_train.csv', train_fraction = 0.7):

        f = open(location, 'rt')
        f.readline()    # Throw away the header line

        self._all_data = [split_line(l)[3:5] for l in f]
        f.close()

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
        non_vanilla_chars = [c for c in self._char_counts if c not in vanilla_chars and self._char_counts[c] >= 10]
        non_vanilla_chars.sort()
        ix_cursor = 0
        for c in vanilla_chars + non_vanilla_chars:
            self._c_to_ix[c] = ix_cursor
            self._ix_to_c[ix_cursor] = c
            ix_cursor += 1
        self._rare_ix = ix_cursor
        for c in self._char_counts:
            if self._char_counts[c] < 10:
                self._c_to_ix[c] = self._rare_ix
        self._vanilla_chars = set(vanilla_chars)    # Used for fast testing
        self._max_nv_chars = 10
        self._stop_ix = len(vanilla_chars) + self._max_nv_chars  # Used in the output to denote the end of the string

    def convert_input_output_pair(self, input_string, output_string):

        # Flag which characters are non-vanilla and assign them indices
        nv_characters = set([c for c in input_string if c not in self._vanilla_chars])
        if len(nv_characters) > self._max_nv_chars:
            raise ValueError('Input string contained {} non-vanilla characters. Maximum set up for is {}.\nInput: {}'.format(len(nv_characters), self._max_nv_chars, input_string))
        nv_char_to_ix = {}
        nv_ix_to_char = {}
        for (c, ix) in zip(nv_characters, np.random.choice(self._max_nv_chars, size=len(nv_characters), replace=False)):
            nv_char_to_ix[c] = ix
            nv_ix_to_char[ix] = c

        # Array to index into embedding array
        input_ix = np.array([self._c_to_ix[c] for c in input_string])
        # One-hot encoding
        input_one_hot = np.zeros(shape=[len(input_string), self._max_nv_chars])
        for (i, c) in enumerate(input_string):
            if c in nv_characters:
                input_one_hot[i, nv_char_to_ix[c]] = 1.0

        # Label
        label_one_hot = np.zeros(shape=[len(output_string)+1, self._stop_ix + 1])
        for (i, c) in enumerate(output_string):
            if c in self._vanilla_chars:
                label_one_hot[i, self._c_to_ix[c]] = 1.0
            else:
                label_one_hot[i, len(self._vanilla_chars) + nv_char_to_ix[c]] = 1.0
        label_one_hot[-1, self._stop_ix] = 1

        return input_ix, input_one_hot, label_one_hot

    def next_training_example(self):

        input_string, output_string = self._all_data[self._training_cursor]

        self._training_cursor = (self._training_cursor + 1) % self._N_train

        return self.convert_input_output_pair(input_string, output_string)


class TestsForTrainingDataset(unittest.TestCase):

    def test_construction_and_draw_example(self):
        train = TrainingDataset()

        input_ix, input_oh, label_oh = train.next_training_example()
        input_ix, input_oh, label_oh = train.next_training_example()
        input_ix, input_oh, label_oh = train.next_training_example()
        input_ix, input_oh, label_oh = train.next_training_example()