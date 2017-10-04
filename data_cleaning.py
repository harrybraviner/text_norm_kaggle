open numpy as np

punctuation = ['.', ',', '/', '\\', '!', '£', '€', '$', '$', '%', '\'', '"', '(', ')']


def is_simple_char(c):
    if ord(c) >= ord('a') and ord(c) <= ord('z'):
        return True
    if ord(c) >= ord('A') and ord(c) <= ord('Z'):
        return True
    if ord(c) >= ord('0') and ord(c) <= ord('9'):
        return True
    if ord(c) in [ord(x) for x in punctuation]: return True

    return False

class TrainingDataset:

    def __init__(self, location = 'data/en_train.csv', train_fraction = 0.7):

        f = open(location, 'rt')
        f.readline()    # Throw away the header line

        self._all_data =
            np.array([(l[3], l[4]) for l in f])
        f.close()

        self._N_total = self._all_data.shape[0]
        self._N_train = int(train_fraction * self_N_total)
        self._N_val = self._N_total - self._N_train

        self._training_data = self._all_data[:self._N_train]
        self._val_data      = self._all_data[self._N_train:]

        self._training_cursor = 0   # Used to keep track of which example to serve next

    def next_training_example(self):

        next_x, next_y = self._training_data[self._training_cursor]
        self._training_cursor = (self._training_cursor + 1) % self._N_train

        return next_x, next_y

