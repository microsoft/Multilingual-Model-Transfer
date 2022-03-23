# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import os
import random

import torch
from torch.utils.data import Dataset

from options import opt

from utils import read_bio_samples

class BioDataset(Dataset):
    """
        tag_vocab is updated
        vocab is updated if update_vocab is True
        self.raw_X, self.raw_Y: list of lists of strings
        self.X: list of dict {'words': list of int, 'chars': list of list of int}
        self.Y: list of lists of integers
    """
    def __init__(self, input_file, vocab, char_vocab, tag_vocab,
                 update_vocab, encoding='utf-8', remove_empty=False):
        self.raw_X = []
        self.raw_Y = []
        with open(input_file, encoding=encoding) as inf:
            for lines in read_bio_samples(inf):
                x = [l.split()[0] for l in lines]
                y = [l.split()[-1] for l in lines]
                if remove_empty:
                    if all([l=='O' for l in y]):
                        continue
                self.raw_X.append(x)
                self.raw_Y.append(y)
                if vocab and update_vocab:
                    for w in x:
                        vocab.add_word(w)
                if char_vocab and update_vocab:
                    for w in x:
                        for ch in w:
                            char_vocab.add_word(ch, normalize=opt.lowercase_char)
                tag_vocab.add_tags(y)
        self.X = []
        self.Y = []
        for xs, ys in zip(self.raw_X, self.raw_Y):
            x = {}
            if vocab:
                x['words'] = [vocab.lookup(w) for w in xs]
            if char_vocab:
                x['chars'] = [[char_vocab.lookup(ch, normalize=opt.lowercase_char) for ch in w] for w in xs]
            self.X.append(x)
            self.Y.append([tag_vocab.lookup(y) for y in ys])
        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def get_subset(self, num_samples, shuffle=True):
        subset = copy.copy(self)
        if shuffle:
            idx = random.sample(range(len(self)), num_samples)
        else:
            idx = list(range(min(len(self), num_samples)))
        subset.raw_X = [subset.raw_X[i] for i in idx]
        subset.raw_Y = [subset.raw_Y[i] for i in idx]
        subset.X = [subset.X[i] for i in idx]
        subset.Y = [subset.Y[i] for i in idx]
        return subset

class ConllDataset(BioDataset):
    def __init__(self, input_file, vocab, char_vocab, tag_vocab,
            update_vocab, remove_empty=False):
        if (input_file[17:20] == 'esp') or (input_file[17:20] == 'ned'):
            encoding = 'ISO-8859-1'
        else:
            encoding = 'utf-8'
        super().__init__(input_file, vocab, char_vocab, tag_vocab, 
                update_vocab, encoding, remove_empty=False)


def get_conll_ner_datasets(vocab, char_vocab, tag_vocab, data_dir, lang):
    print(f'Loading CoNLL NER data for {lang} Language..')
    train_set = ConllDataset(os.path.join(data_dir, f'{lang}.train'),
                           vocab, char_vocab, tag_vocab, update_vocab=True, remove_empty=opt.remove_empty_samples)
    dev_set = ConllDataset(os.path.join(data_dir, f'{lang}.dev'),
                           vocab, char_vocab, tag_vocab, update_vocab=True)
    test_set = ConllDataset(os.path.join(data_dir, f'{lang}.test'),
                           vocab, char_vocab, tag_vocab, update_vocab=True)
    return train_set, dev_set, test_set, train_set


def get_train_on_translation_conll_ner_datasets(vocab, char_vocab, tag_vocab, data_dir, lang):
    print(f'Loading Train-on-Translation CoNLL NER data for {lang} Language..')
    train_set = BioDataset(os.path.join(data_dir, f'eng2{lang}_{lang}.train'),
                           vocab, char_vocab, tag_vocab, update_vocab=True)
    dev_set = BioDataset(os.path.join(data_dir, f'{lang}.dev'),
                           vocab, char_vocab, tag_vocab, update_vocab=True)
    test_set = BioDataset(os.path.join(data_dir, f'{lang}.test'),
                           vocab, char_vocab, tag_vocab, update_vocab=False)
    return train_set, dev_set, test_set, train_set


def get_test_on_translation_conll_ner_datasets(vocab, char_vocab, tag_vocab, data_dir, lang):
    print(f'Loading Test-on-Translation CoNLL NER data for {lang} Language..')
    train_set = BioDataset(os.path.join(data_dir, f'eng.train'),
                           vocab, char_vocab, tag_vocab, update_vocab=True)
    dev_set = BioDataset(os.path.join(data_dir, f'eng2{lang}_eng.dev'),
                           vocab, char_vocab, tag_vocab, update_vocab=True)
    test_set = BioDataset(os.path.join(data_dir, f'eng2{lang}_eng.test'),
                           vocab, char_vocab, tag_vocab, update_vocab=False)
    return train_set, dev_set, test_set, train_set
