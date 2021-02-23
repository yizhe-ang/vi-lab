import torch


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq, alphabet, seq):
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq]
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X
