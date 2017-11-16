# Usage

Download the training data for the [Text Normalization Challenge](https://www.kaggle.com/c/text-normalization-challenge-english-language) and place the file `en_train.csv` into the `data` subdirectory.

Run the command
```
./norm_net.py
```
to train a model.
The flag `--mini_dataset` can be added to use only the first 1000 entries in the training file.
This is useful when making changes to the code and checking that it still runs (without having to wait while the very large training dataset is processed).

# Data cleaning

The training set contains a large number of distinct characters (3080).
Many of these are CJK characters which are outputted verbatim.
Many more are symbols that are translated into latin characters (e.g. 'alpha').

I don't want to have a large number of outputs in the final layer, but I also
don't want to prevent the system from repeating a character from the input.
Therefore I have the following idea.

We will define a set of 'vanilla characters'. These will be a-z, A-Z, and 0-9, space, and the quotation mark.
(These are chosen since they are able to appear in the output without also appearing in the input.
The same is true of e-acute, but only when the input is 'Pate'.)
There are no more than 8 distinct non-vanilla characters in any one input in the training set.
Any character that occurs fewer than 10 times in the training set is a 'rare' character.
These will all be treated as a single `<RARE>` token.
The remaining characters (those that are not alphanumeric, but have 10 or more occurrances)
will each get their own embedding, but will not be available directly for output.

Processing an input will take place as follows:
* Form the set of distinct non-vanilla characters for the input.
* Assign distinct, randomized numbers from 0-9 to these characters.
* Replace any rare characters with the `<RARE>` token.
* Convert each character to its embedded representation.
* Pass the embeddings, and the one-hot encodings of the 0-9 indices, to the network.
* For the output, one-hot encode the characters appropriately.

# Network architecture

The network itself will be two LSTM neural nets:
The *encoder* takes the input string (suitably encoded). The outputs of the net are ignored, we are only interested in its final state.
The final state of the encoder is fed to the *decoder*, a second LSTM net. The input of each step of the decoder is the output character of the previous step.
The decoder is trained to produce the output string.
Parameters are:
* `embedding_size` - dimension of the dense embedding of characters
* `layer_width_1` - number of hidden neurons in the first layer
* `layer_width_2` - ditto for seocnd layer

The input will be the dense embedded encoding of the character (or the embedding of `<RARE>` if the character is not common in the input dataset) concatenated with the one-hot encoding of the non-vanilla index.

# Todo

* Convert to a more object-oriented architecture.
* Loading the test set? Should be made part of the TrainingDataset object (since it wants access to
  the indices of the non-vanilla characters)?
