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
        self._validation_cursor = self._N_train   # Used to keep track of which example to serve next

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

    def is_vanilla(self, c):
        return (c in self._vanilla_chars)

    @property
    def num_non_rare_chars(self):
        return len(self._ix_to_c)

    @property
    def num_vanilla_chars(self):
        return len(self._vanilla_chars)

    def get_index_of_char(self, c):
        return self._c_to_ix[c]
    
    def next_training_example(self):

        input_string, output_string = self._all_data[self._training_cursor]

        self._training_cursor = (self._training_cursor + 1) % self._N_train

        return input_string, output_string
    
    def next_batch(self, batch_size):

        input_output_strings = [self.next_training_example() for _ in range(batch_size)]

#        input_ix = np.stack([x[0] for x in input_output_data])
#        input_oh = np.stack([x[1] for x in input_output_data])
#        output_y = np.stack([x[2] for x in input_output_data])
#        input_length  = np.stack([x[3] for x in input_output_data])
#        output_length = np.stack([x[4] for x in input_output_data])

        return input_output_strings
    
    def get_validation_set(self, batch_size):

        done = False
        examples_served = 0
        while not done:
            if (self._N_val - examples_served > batch_size):
                batch = [self._all_data[self._N_train + examples_served + i] for i in range(batch_size)]
                examples_served += batch_size
                yield batch
            else:
                num_to_serve = self._N_val - examples_served
                batch = [self._all_data[self._N_train + examples_served + i] for i in range(num_to_serve)]
                examples_served += num_to_serve
                done = True
                yield batch

            
        



class TestsForTrainingDataset(unittest.TestCase):

    def test_construction_and_draw_example(self):
        train = TrainingDataset(line_limit = 1000)

        _, _, _, _, _ = train.next_training_example()
        _, _, _, _, _ = train.next_training_example()
        _, _, _, _, _ = train.next_training_example()
        _, _, _, _, _ = train.next_training_example()
