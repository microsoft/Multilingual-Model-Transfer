import glob
import os
import random

import torch
from torch.utils.data import Dataset

from options import opt


class MultiLangAmazonDataset(Dataset):
    """
        vocab is updated if update_vocab is True
        self.raw_X: list of lists of strings
        self.X, self.Y: list of lists of integers
    """
    num_labels = 2
    def __init__(self, input_file=None, vocab=None, char_vocab=None, max_seq_len=0, num_train_lines=0, update_vocab=False):
        self.raw_X = []
        self.X = []
        self.Y = []
        if input_file is None:
            return
        with open(input_file) as inf:
            for line in inf:
                parts = line.split('\t')
                assert len(parts) == 2, f"Incorrect format {line}"
                x = parts[1].rstrip().split(' ')
                if max_seq_len > 0:
                    x = x[:max_seq_len]
                if update_vocab:
                    for w in x:
                        vocab.add_word(w)
                if char_vocab and update_vocab:
                    for w in x:
                        for ch in w:
                            char_vocab.add_word(ch, normalize=opt.lowercase_char)
                self.raw_X.append(x)
                sample = {}
                if vocab:
                    sample['words'] = [vocab.lookup(w) for w in x]
                if char_vocab:
                    sample['chars'] = [[char_vocab.lookup(ch,
                            normalize=opt.lowercase_char) for ch in w] for w in x]
                self.X.append(sample)
                self.Y.append(int(parts[0]))
        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def get_subset(self, num_samples, shuffle=True):
        subset = MultiLangAmazonDataset()
        if shuffle:
            idx = random.sample(range(len(self)), num_samples)
        else:
            idx = list(range(min(len(self), num_samples)))
        subset.raw_X = [subset.raw_X[i] for i in idx]
        subset.raw_Y = [subset.raw_Y[i] for i in idx]
        subset.X = [subset.X[i] for i in idx]
        subset.Y = [subset.Y[i] for i in idx]
        return subset

    def train_dev_split(self, dev_samples, shuffle=True):
        devset = MultiLangAmazonDataset()
        idx = list(range(len(self)))
        if shuffle:
            random.shuffle(idx)
        # dev set
        dev_idx = idx[:dev_samples]
        devset.raw_X = [self.raw_X[i] for i in dev_idx]
        devset.X = [self.X[i] for i in dev_idx]
        devset.Y = [self.Y[i] for i in dev_idx]
        # update train set
        train_idx = idx[dev_samples:]
        self.raw_X = [self.raw_X[i] for i in train_idx]
        self.X = [self.X[i] for i in train_idx]
        self.Y = [self.Y[i] for i in train_idx]
        self.max_seq_len = None
        return devset

    def get_max_seq_len(self):
        if not hasattr(self, 'max_seq_len') or self.max_seq_len is None:
            self.max_seq_len = max([len(x) for x in self.X])
        return self.max_seq_len


def get_multi_lingual_amazon_datasets(vocab, char_vocab, root_dir, domain, lang, max_seq_len):
    print(f'Loading Multi-Lingual Amazon Review data for {domain} Domain and {lang} Language..')
    data_dir = os.path.join(root_dir, lang, domain)

    train_set = MultiLangAmazonDataset(os.path.join(data_dir, 'train.tok.txt'),
                           vocab, char_vocab, max_seq_len, 0, update_vocab=True)
    # train-dev split
    dev_set = train_set.train_dev_split(400, shuffle=True)
    test_set = MultiLangAmazonDataset(os.path.join(data_dir, 'test.tok.txt'),
                           vocab, char_vocab, max_seq_len, 0, update_vocab=True)
    unlabeled_set = MultiLangAmazonDataset(os.path.join(data_dir, 'unlabeled.tok.txt'),
                           vocab, char_vocab, max_seq_len, 50000, update_vocab=True)
    return train_set, dev_set, test_set, unlabeled_set
