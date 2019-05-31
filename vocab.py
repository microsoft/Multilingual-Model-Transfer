# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn

from options import opt


class Vocab:
    def __init__(self, emb_size, txt_file=None):
        self.vocab_size, self.emb_size = 0, emb_size
        self.embeddings = []
        self.w2vvocab = {}
        self.v2wvocab = []
        # load pretrained embedings
        if txt_file:
            with open(txt_file, 'r') as inf:
                parts = inf.readline().split()
                assert len(parts) == 2
                vs, es = int(parts[0]), int(parts[1])
                assert es == self.emb_size
                # add an UNK token
                self.pretrained = np.empty((vs, es), dtype=np.float)
                self.pt_v2wvocab = []
                self.pt_w2vvocab = {}
                cnt = 0
                for line in inf.readlines():
                    parts = line.rstrip().split(' ')
                    word = parts[0]
                    # add to vocab
                    self.pt_v2wvocab.append(word)
                    self.pt_w2vvocab[word] = cnt
                    # load vector
                    vector = [float(x) for x in parts[-self.emb_size:]]
                    self.pretrained[cnt] = vector
                    cnt += 1
        # add EOS token
        self.eos_tok = '</s>'
        self.add_word(self.eos_tok)
        opt.eos_idx = self.eos_idx = self.w2vvocab[self.eos_tok]
        self.embeddings[self.eos_idx][:] = 0
        # add <unk>
        self.unk_tok = '<unk>'
        self.add_word(self.unk_tok)
        self.unk_idx = self.w2vvocab[self.unk_tok]

    def base_form(word):
        return word.strip().lower()

    def new_rand_emb(self):
        vec = np.random.normal(0, 1, size=self.emb_size)
        vec /= sum(x*x for x in vec) ** .5
        return vec

    def init_embed_layer(self, fix_emb):
        # free some memory
        self.clear_pretrained_vectors()
        emb = nn.Embedding.from_pretrained(torch.tensor(self.embeddings, dtype=torch.float),
                freeze=fix_emb)
        emb.padding_idx = self.eos_idx
        assert len(emb.weight) == self.vocab_size
        return emb

    def add_word(self, word, normalize=True):
        if normalize:
            word = Vocab.base_form(word)
        if word not in self.w2vvocab:
            if not opt.random_emb and hasattr(self, 'pt_w2vvocab') and word in self.pt_w2vvocab:
                vector = self.pretrained[self.pt_w2vvocab[word]].copy()
            else:
                vector = self.new_rand_emb()
            self.v2wvocab.append(word)
            self.w2vvocab[word] = self.vocab_size
            self.embeddings.append(vector)
            self.vocab_size += 1

    def clear_pretrained_vectors(self):
        if hasattr(self, 'pretrained'):
            del self.pretrained
            del self.pt_w2vvocab
            del self.pt_v2wvocab

    def lookup(self, word, normalize=True):
        if normalize:
            word = Vocab.base_form(word)
        if word in self.w2vvocab:
            return self.w2vvocab[word]
        return self.unk_idx

    def get_word(self, i):
        return self.v2wvocab[i]


class TagVocab:
    def __init__(self, bio=True):
        self.tag2id = {}
        self.id2tag = []
        self.bio = bio

    def __len__(self):
        return len(self.tag2id)

    def _add_tag(self, tag):
        if tag not in self.tag2id:
            self.tag2id[tag] = len(self)
            self.id2tag.append(tag)

    def add_tag(self, tag):
        self._add_tag(tag)
        if self.bio:
            if tag.startswith('B-') or tag.startswith('I-'):
                self._add_tag(f'B-{tag[2:]}')
                self._add_tag(f'I-{tag[2:]}')

    def add_tags(self, tags):
        for tag in tags:
            self.add_tag(tag)

    def lookup(self, raw_y):
        return self.tag2id[raw_y]

    def get_tag(self, i):
        return self.id2tag[i]

