"""Defines Moltokenizer class to encode and decode strings."""
from collections import OrderedDict

__author__ = 'Bonggun Shin'

class Moltokenizer(object):
    """Encodes and decodes strings to/from integer IDs."""

    def __init__(self, vocab_file, model_version="v1"):
        """Initializes class, creating a vocab file if data_files is provided."""
        print("Initializing Moltokenizer from file %s." %
                                  vocab_file)

        self.vocab, self.token_list = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.cache = {}
        self.model_version = model_version

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = OrderedDict()
        token_list = []
        index = 0
        with open(vocab_file, "rt") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                token_list.append(token)
                index += 1
        return vocab, token_list

    def tokenize(self, text):
        tokens = []
        if self.model_version=="v2":
            tokens.append("[CLS]")
        tokens.append("[BEGIN]")
        for c in text:
            tokens.append(c)
        tokens.append("[END]")

        return tokens

    def encode(self, sequence):
        if sequence not in self.cache:
            tokens = self.tokenize(sequence)
            self.cache[sequence] = self.convert_by_vocab(self.vocab, tokens)

        return self.cache[sequence]

    def decode(self, ids):
        trimed_ids = [token for token in ids if token!=0] # delete all [PAD]'s
        if len(trimed_ids)==0:
            return ''
        # if trimed_ids[0]!=1 or trimed_ids[-1]!=2:  # invalid smiles then return empty string
        if self.model_version == "v2":
            if trimed_ids[0] != 1 or trimed_ids[1] != 2 or trimed_ids[-1] != 3:  # invalid smiles then return empty string
                return ''
            tokens = self.convert_by_vocab(self.inv_vocab, trimed_ids[2:-1])
        else:
            if trimed_ids[0] != 1 or trimed_ids[-1] != 2:  # invalid smiles then return empty string
                return ''
            tokens = self.convert_by_vocab(self.inv_vocab, trimed_ids[1:-1])

        return ''.join(tokens)

    def convert_by_vocab(self, vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

if __name__ == "__main__":
    vocab_file = '../../../data/optgen_vocab.txt'

    moltokenizer = Moltokenizer(vocab_file)

    ids = moltokenizer.encode('C=CC(=O)C')
    print(ids)
    smiles = moltokenizer.decode(ids)
    print(smiles)

