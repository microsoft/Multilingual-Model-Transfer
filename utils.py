# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import subprocess

import numpy as np
import torch
from torch import autograd
from options import opt


def read_bio_samples(f):
    lines = []
    for line in f:
        if not line.rstrip():
            if len(lines) > 0:
                yield lines
            lines = []
        else:
            lines.append(line.rstrip())
    if len(lines) > 0:
        yield lines
        lines = []

def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True


def sorted_collate(batch):
    return my_collate(batch, sort=True)


def unsorted_collate(batch):
    return my_collate(batch, sort=False)


def my_collate(batch, sort):
    x, y = zip(*batch)
    # extract input indices
    if opt.default_emb == 'elmo':
        x, y = raw_pad(x, y, sort)
    else:
        x, y = pad(x, y, opt.eos_idx, sort)
    return (x, y)


def sorted_cls_collate(batch):
    return cls_my_collate(batch, sort=True)


def unsorted_cls_collate(batch):
    return cls_my_collate(batch, sort=False)


def cls_my_collate(batch, sort):
    x, y = zip(*batch)
    # extract input indices
    x, y = cls_pad(x, y, opt.eos_idx, sort)
    return (x, y)


# TODO do not actually prepare input if char emb is not used to speed up
def pad(x, y, eos_idx, sort):
    bs = len(x)
    chars = [sample['chars'] for sample in x]
    x = [sample['words'] for sample in x]
    lengths = [len(row) for row in x] if x else [len(row) for row in chars]
    max_len = max(lengths)
    # if using CNN, pad to at least the largest kernel size
    if opt.model.lower() == 'cnn' or opt.D_model.lower() == 'cnn':
        max_len = max(max_len, opt.max_kernel_size)
    if chars:
        char_lengths = [[len(w) for w in sample] for sample in chars]
        max_char_len = max([l for sample in char_lengths for l in sample])
        if opt.charemb_model.lower() == 'cnn':
            max_char_len = max(max_char_len, max(opt.charemb_kernel_sizes))
    # pad sequences
    lengths = torch.tensor(lengths)
    padded_x = torch.full((len(x), max_len), eos_idx, dtype=torch.long)
    padded_y = torch.full((len(y), max_len), -1, dtype=torch.long)
    if chars:
        padded_chars = torch.full((bs, max_len, max_char_len), eos_idx, dtype=torch.long)
        padded_char_lengths = torch.zeros((len(x), max_len), dtype=torch.long)
    for i, (row, tag, char_row, cl_row) in enumerate(zip(x, y, chars, char_lengths)):
        assert len(row) == len(tag)
        assert eos_idx not in row, f'EOS in sequence {row}'
        padded_x[i][:len(row)] = torch.tensor(row)
        padded_y[i][:len(tag)] = torch.tensor(tag)
        if chars:
            padded_char_lengths[i][:len(cl_row)] = torch.tensor(cl_row)
            for j, ch_word in enumerate(char_row):
                padded_chars[i][j][:len(ch_word)] = torch.tensor(ch_word)
    # create mask
    idxes = torch.arange(0, max_len, dtype=torch.long).unsqueeze(0) # some day, you'll be able to directly do this on cuda
    mask = (idxes<lengths.unsqueeze(1)).float()
    if sort:
        # sort by length
        lengths, sort_idx = lengths.sort(0, descending=True)
        padded_x = padded_x.index_select(0, sort_idx)
        padded_y = padded_y.index_select(0, sort_idx)
        if chars:
            padded_chars = padded_chars.index_select(0, sort_idx)
            padded_char_lengths = padded_char_lengths.index_select(0, sort_idx)
        mask = mask.index_select(0, sort_idx)
    padded_x = padded_x.to(opt.device)
    padded_y = padded_y.to(opt.device)
    mask = mask.to(opt.device)
    if chars:
        padded_chars = padded_chars.to(opt.device)
        padded_char_lengths = padded_char_lengths.to(opt.device)
    else:
        padded_chars = padded_char_lengths = None
    return (padded_x, lengths, mask, padded_chars, padded_char_lengths), padded_y


def raw_pad(x, y, sort):
    bs = len(x)
    lengths = [len(row) for row in x]
    max_len = max(lengths)
    # if using CNN, pad to at least the largest kernel size
    if opt.model.lower() == 'cnn' or opt.D_model.lower() == 'cnn':
        max_len = max(max_len, opt.max_kernel_size)
    # pad sequences
    lengths = torch.tensor(lengths)
    padded_y = torch.full((len(y), max_len), -1, dtype=torch.long)
    for i, tag in enumerate(y):
        padded_y[i][:len(tag)] = torch.tensor(tag)
    # create mask
    idxes = torch.arange(0, max_len, dtype=torch.long).unsqueeze(0) # some day, you'll be able to directly do this on cuda
    mask = (idxes<lengths.unsqueeze(1)).float()
    if sort:
        # sort by length
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = [x[i] for i in sort_idx]
        padded_y = padded_y.index_select(0, sort_idx)
        mask = mask.index_select(0, sort_idx)
    padded_y = padded_y.to(opt.device)
    mask = mask.to(opt.device)
    return (x, lengths, mask), padded_y



def cls_pad(x, y, eos_idx, sort):
    bs = len(x)
    chars = [sample['chars'] for sample in x] if opt.use_charemb else None
    x = [sample['words'] for sample in x]
    lengths = [len(row) for row in x]
    max_len = max(lengths)
    # if using CNN, pad to at least the largest kernel size
    if opt.model.lower() == 'cnn':
        max_len = max(max_len, opt.max_kernel_size)
    if chars:
        char_lengths = [[len(w) for w in sample] for sample in chars]
        max_char_len = max([l for sample in char_lengths for l in sample])
        if opt.charemb_model.lower() == 'cnn':
            max_char_len = max(max_char_len, max(opt.charemb_kernel_sizes))
    # pad sequences
    lengths = torch.tensor(lengths)
    y = torch.LongTensor(y).view(-1)
    padded_x = torch.full((len(x), max_len), eos_idx, dtype=torch.long)
    if chars:
        padded_chars = torch.full((bs, max_len, max_char_len), eos_idx, dtype=torch.long)
        padded_char_lengths = torch.zeros((len(x), max_len), dtype=torch.long)
        for i, (row, char_row, cl_row) in enumerate(zip(x, chars, char_lengths)):
            assert eos_idx not in row, f'EOS in sequence {row}'
            padded_x[i][:len(row)] = torch.tensor(row)
            padded_char_lengths[i][:len(cl_row)] = torch.tensor(cl_row)
            for j, ch_word in enumerate(char_row):
                padded_chars[i][j][:len(ch_word)] = torch.tensor(ch_word)
    else:
        for i, row in enumerate(x):
            assert eos_idx not in row, f'EOS in sequence {row}'
            padded_x[i][:len(row)] = torch.tensor(row)
    # create mask
    idxes = torch.arange(0, max_len, dtype=torch.long).unsqueeze(0) # some day, you'll be able to directly do this on cuda
    mask = (idxes<lengths.unsqueeze(1)).float()
    if sort:
        # sort by length
        lengths, sort_idx = lengths.sort(0, descending=True)
        padded_x = padded_x.index_select(0, sort_idx)
        y = y.index_select(0, sort_idx)
        if chars:
            padded_chars = padded_chars.index_select(0, sort_idx)
            padded_char_lengths = padded_char_lengths.index_select(0, sort_idx)
        mask = mask.index_select(0, sort_idx)
    padded_x = padded_x.to(opt.device)
    y = y.to(opt.device)
    mask = mask.to(opt.device)
    if chars:
        padded_chars = padded_chars.to(opt.device)
        padded_char_lengths = padded_char_lengths.to(opt.device)
        return (padded_x, lengths, mask, padded_chars, padded_char_lengths), y
    else:
        padded_chars = padded_char_lengths = None
        return (padded_x, lengths, mask, None, None), y


def calc_gradient_penalty(D, features, onesided=False, interpolate=True):
    feature_vecs = list(features.values())

    if interpolate:
        alpha = torch.rand(feature_vecs[0].size())
        alpha /= torch.sum(alpha, dim=1, keepdim=True)
        alpha = alpha.cuda() if opt.use_cuda else alpha
        interpolates = sum([f*alpha[i] for (i,f) in enumerate(feature_vecs)])
    else:
        feature = torch.cat(feature_vecs, dim=0)
        alpha = torch.rand(len(feature), 1).expand(feature.size())
        noise = torch.rand(feature.size())
        alpha = alpha.cuda() if opt.use_cuda else alpha
        noise = noise.cuda() if opt.use_cuda else noise
        interpolates = alpha*feature + (1-alpha)*(feature+0.5*feature.std()*noise)

    interpolates = interpolates.to(opt.device, require_grad=True)
    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                    if opt.use_cuda else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    if onesided:
        clip_fn = lambda x: x.clamp(min=0)
    else:
        clip_fn = lambda x: x
    gradient_penalty = (clip_fn(gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.gp_lambd
    return gradient_penalty


def calc_orthogality_loss(shared_features, lang_features):
    assert len(shared_features) == len(lang_features)
    loss = None
    for i, sf in enumerate(shared_features):
        df = lang_features[i]
        prod = torch.mm(sf.t(), df)
        loss = torch.sum(prod*prod) + (loss if loss is not None else 0)
    return loss


def average_cv_accuracy(cv):
    """
    cv[fold]['valid'] contains CV accuracy for validation set
    cv[fold]['test'] contains CV accuracy for test set
    """
    avg_acc = {'valid': defaultdict(float), 'test': defaultdict(float)}
    for fold, foldacc in cv.items():
        for dataset, cv_acc in foldacc.items():
            for lang, acc in cv_acc.items():
                avg_acc[dataset][lang] += acc
    for lang in avg_acc['valid']:
        avg_acc['valid'][lang] /= opt.kfold
        avg_acc['test'][lang] /= opt.kfold
    # overall average
    return avg_acc


def endless_get_next_batch(loaders, iters, lang):
    try:
        inputs, targets = next(iters[lang])
    except StopIteration:
        iters[lang] = iter(loaders[lang])
        inputs, targets = next(iters[lang])
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size == 1
    # if opt.skip_leftover_batch and len(targets) < opt.batch_size:
    #     return endless_get_next_batch(loaders, iters, lang)
    return (inputs, targets)


lang_labels = {}
def get_lang_label(loss, lang, size):
    if (lang, size) in lang_labels:
        return lang_labels[(lang, size)]
    idx = opt.all_langs.index(lang)
    if loss.lower() == 'l2':
        labels = torch.FloatTensor(size, len(opt.all_langs))
        labels.fill_(-1)
        labels[:, idx].fill_(1)
    else:
        labels = torch.LongTensor(size)
        labels.fill_(idx)
    labels = labels.to(opt.device)
    lang_labels[(lang, size)] = labels
    return labels


random_lang_labels = {}
def get_random_lang_label(loss, size):
    if size in random_lang_labels:
        return random_lang_labels[size]
    labels = torch.FloatTensor(size, len(opt.all_langs))
    if loss.lower() == 'l2':
        labels.fill_(0)
    else:
        labels.fill_(1 / len(opt.all_langs))
    labels = labels.to(opt.device)
    random_lang_labels[size] = labels
    return labels


def get_gate_label(gate_out, lang, mask, expert_sp, all_langs=False):
    langs = opt.all_langs if all_langs else opt.langs
    num_experts, idx = len(langs), langs.index(lang)
    if expert_sp: # 0 is the shared gate
        idx += 1
        num_experts += 1
        labels = torch.zeros_like(gate_out, dtype=torch.float)
        labels[:, :, idx].fill_(1.)
        if expert_sp:
            labels[:, :, 0].fill_(1.)
        labels = labels * mask.unsqueeze(-1)
    else:
        labels = torch.full(gate_out.size()[:-1], idx, dtype=torch.long).to(opt.device)
        labels = labels * mask.long().expand_as(labels) - (1 - mask.long().expand_as(labels))
    labels = labels.to(opt.device)
    return labels


def get_cls_gate_label(gate_out, lang, expert_sp):
    num_experts, idx = len(opt.langs), opt.langs.index(lang)
    if expert_sp: # 0 is the shared gate
        idx += 1
        num_experts += 1
        labels = torch.zeros_like(gate_out, dtype=torch.float)
        labels[:, idx].fill_(1.)
        if expert_sp:
            labels[:, 0].fill_(1.)
    else:
        labels = torch.full(gate_out.size()[:-1], idx, dtype=torch.long)
    labels = labels.to(opt.device)
    return labels


def conllF1(stream, log):
    stream.seek(0)
    out = subprocess.check_output('./conlleval.pl', input=stream.getvalue(), encoding='utf-8')
    out = out.splitlines()
    if log:
        log.info(out[1])
    f1 = float(out[1].split()[-1])
    return f1


def gmean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1. / len(a))


def get_padding_size(kernel_size):
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    # only pad top and down
    return (0, 0, ka, kb)
