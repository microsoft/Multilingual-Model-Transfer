import pdb
from collections import defaultdict
import io
import itertools
import logging
import math
import os
import pickle
import random
import shutil
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from options import opt
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

from data_prep.bio_dataset import *
from models import *
import utils
from vocab import Vocab, TagVocab

# save models and logging
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
# output options
log.info(opt)


def train(vocabs, char_vocab, tag_vocab, train_sets, dev_sets, test_sets, unlabeled_sets):
    """
    train_sets, dev_sets, test_sets: dict[lang] -> AmazonDataset
    For unlabeled langs, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters, d_unlabeled_iters = {}, {}, {}
    dev_loaders, test_loaders = {}, {}
    my_collate = utils.sorted_collate if opt.model=='lstm' else utils.unsorted_collate
    for lang in opt.langs:
        train_loaders[lang] = DataLoader(train_sets[lang],
                opt.batch_size, shuffle=True, collate_fn = my_collate)
        train_iters[lang] = iter(train_loaders[lang])
    for lang in opt.dev_langs:
        dev_loaders[lang] = DataLoader(dev_sets[lang],
                opt.batch_size, shuffle=False, collate_fn = my_collate)
        test_loaders[lang] = DataLoader(test_sets[lang],
                opt.batch_size, shuffle=False, collate_fn = my_collate)
    for lang in opt.all_langs:
        if lang in opt.unlabeled_langs:
            uset = unlabeled_sets[lang]
        else:
            # for labeled langs, consider which data to use as unlabeled set
            if opt.unlabeled_data == 'both':
                uset = ConcatDataset([train_sets[lang], unlabeled_sets[lang]])
            elif opt.unlabeled_data == 'unlabeled':
                uset = unlabeled_sets[lang]
            elif opt.unlabeled_data == 'train':
                uset = train_sets[lang]
            else:
                raise Exception(f'Unknown options for the unlabeled data usage: {opt.unlabeled_data}')
        unlabeled_loaders[lang] = DataLoader(uset,
                opt.batch_size, shuffle=True, collate_fn = my_collate)
        unlabeled_iters[lang] = iter(unlabeled_loaders[lang])
        d_unlabeled_iters[lang] = iter(unlabeled_loaders[lang])

    # embeddings
    emb = MultiLangWordEmb(vocabs, char_vocab, opt.use_wordemb, opt.use_charemb).to(opt.device)
    # models
    F_s = None
    F_p = {}
    C, D = None, None
    if opt.model.lower() == 'lstm':
        if opt.shared_hidden_size > 0:
            F_s = LSTMFeatureExtractor(opt.total_emb_size, opt.F_layers, opt.shared_hidden_size,
                                       opt.dropout, opt.bdrnn)
        if opt.private_hidden_size > 0:
            if not opt.concat_sp:
                assert opt.shared_hidden_size == opt.private_hidden_size, "shared dim != private dim when using add_sp!"
            for lang in opt.langs:
                F_p[lang] = LSTMFeatureExtractor(opt.total_emb_size, opt.F_layers, opt.private_hidden_size,
                                                   opt.dropout, opt.bdrnn)
    elif opt.model.lower() == 'cnn':
        # TODO
        raise NotImplemented()
        if opt.shared_hidden_size > 0:
            F_s = CNNFeatureExtractor(vocab, opt.F_layers, opt.shared_hidden_size,
                                      opt.kernel_num, opt.kernel_sizes, opt.dropout)
        if opt.private_hidden_size > 0:
            for lang in opt.langs:
                F_p[lang] = CNNFeatureExtractor(vocab, opt.F_layers, opt.private_hidden_size,
                                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise Exception(f'Unknown model architecture {opt.model}')

    C = SpMlpTagger(opt.C_layers, opt.shared_hidden_size, opt.private_hidden_size, opt.concat_sp,
            opt.shared_hidden_size + opt.private_hidden_size, len(tag_vocab),
            opt.mlp_dropout, opt.C_bn)
    if opt.shared_hidden_size > 0 and opt.n_critic > 0:
        if opt.D_model.lower() == 'lstm':
            d_args = {
                'num_layers': opt.D_lstm_layers,
                'input_size': opt.shared_hidden_size,
                'hidden_size': opt.shared_hidden_size,
                'dropout': opt.D_dropout,
                'bdrnn': opt.D_bdrnn,
                'attn_type': opt.D_attn
            }
        elif opt.D_model.lower() == 'cnn':
            d_args = {
                'num_layers': 1,
                'input_size': opt.shared_hidden_size,
                'hidden_size': opt.shared_hidden_size,
                'kernel_num': opt.D_kernel_num,
                'kernel_sizes': opt.D_kernel_sizes,
                'dropout': opt.D_dropout
            }
        else:
            d_args = None

        D = LanguageDiscriminator(opt.D_model, opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
                len(opt.all_langs), opt.D_dropout, opt.D_bn, d_args)
    
    F_s, C, D = F_s.to(opt.device) if F_s else None, C.to(opt.device), D.to(opt.device) if D else None
    if F_p:
        for f_p in F_p.values():
            f_p = f_p.to(opt.device)
    # optimizers
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(*map(list,
        [emb.parameters(), F_s.parameters() if F_s else [], C.parameters()] + \
        [f.parameters() for f in F_p.values()]))),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay)
    if D:
        optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate, weight_decay=opt.D_weight_decay)

    # testing
    if opt.test_only:
        log.info(f'Loading model from {opt.model_save_file}...')
        if F_s:
            F_s.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                f'netF_s.pth')))
        for lang in opt.all_langs:
            if lang in F_p:
                F_p[lang].load_state_dict(torch.load(os.path.join(opt.model_save_file,
                    f'net_F_p_{lang}.pth')))
        C.load_state_dict(torch.load(os.path.join(opt.model_save_file,
            f'netC.pth')))
        if D:
            D.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                f'netD.pth')))

        log.info('Evaluating validation sets:')
        acc = {}
        for lang in opt.all_langs:
            acc[lang] = evaluate(f'{lang}_dev', dev_loaders[lang], vocabs[lang], tag_vocab,
                    emb, lang, F_s, F_p[lang] if lang in F_p else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        test_acc = {}
        for lang in opt.all_langs:
            test_acc[lang] = evaluate(f'{lang}_test', test_loaders[lang], vocabs[lang], tag_vocab,
                    emb, lang, F_s, F_p[lang] if lang in F_p else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average test accuracy: {avg_test_acc}')
        return {'valid': acc, 'test': test_acc}

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0
    # lambda scheduling
    if opt.lambd > 0 and opt.lambd_schedule:
        opt.lambd_orig = opt.lambd
    # adapt max_epoch
    num_iter = int(utils.gmean([len(train_loaders[l]) for l in opt.langs]))
    if opt.max_epoch > 0 and num_iter * opt.max_epoch < 15000:
        opt.max_epoch = 15000 // num_iter
        log.info(f"Setting max_epoch to {opt.max_epoch}")
    for epoch in range(opt.max_epoch):
        emb.train()
        if F_s:
            F_s.train()
        C.train()
        if D:
            D.train()
        if F_p:
            for f in F_p.values():
                f.train()
            
        # lambda scheduling
        if hasattr(opt, 'lambd_orig') and opt.lambd_schedule:
            if epoch == 0:
                opt.lambd = opt.lambd_orig
            elif epoch == 5:
                opt.lambd = 10 * opt.lambd_orig
            elif epoch == 15:
                opt.lambd = 100 * opt.lambd_orig
            log.info(f'Scheduling lambda = {opt.lambd}')

        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        d_correct, d_total = 0, 0
        # conceptually view 1 epoch as 1 epoch of the first lang
        # num_iter = len(train_loaders[opt.langs[0]])
        # num_iter = int(utils.gmean([len(train_loaders[l]) for l in opt.langs]))
        for i in tqdm(range(num_iter), ascii=True):
            # D iterations
            if opt.shared_hidden_size > 0:
                utils.freeze_net(emb)
                utils.freeze_net(F_s)
                map(utils.freeze_net, F_p.values())
                utils.freeze_net(C)
                utils.unfreeze_net(D)
                # WGAN n_critic trick since D trains slower
                n_critic = opt.n_critic
                if opt.wgan_trick:
                    if opt.n_critic>0 and ((epoch==0 and i<25) or i%500==0):
                        n_critic = 100

                for _ in range(n_critic):
                    D.zero_grad()
                    loss_d = {}
                    lang_features = {}
                    # train on both labeled and unlabeled langs
                    for lang in opt.all_langs:
                        # targets not used
                        d_inputs, _ = utils.endless_get_next_batch(
                                unlabeled_loaders, d_unlabeled_iters, lang)
                        d_inputs, d_lengths, _, d_chars, d_char_lengths = d_inputs
                        d_targets = utils.get_lang_label(opt.loss, lang, len(d_lengths))
                        d_embeds = emb(lang, d_inputs, d_chars, d_char_lengths)
                        shared_feat = F_s((d_embeds, d_lengths))
                        if opt.grad_penalty != 'none':
                            lang_features[lang] = shared_feat.detach()
                        d_outputs = D((shared_feat, d_lengths))
                        # D accuracy
                        _, pred = torch.max(d_outputs, 1)
                        d_total += len(d_lengths)
                        if opt.loss.lower() == 'l2':
                            _, tgt_indices = torch.max(d_targets, 1)
                            d_correct += (pred==tgt_indices).sum().item()
                            l_d = functional.mse_loss(d_outputs, d_targets)
                            l_d.backward()
                        else:
                            d_correct += (pred==d_targets).sum().item()
                            l_d = functional.nll_loss(d_outputs, d_targets)
                            l_d.backward()
                        loss_d[lang] = l_d.item()
                    # gradient penalty
                    if opt.grad_penalty != 'none':
                        gp = utils.calc_gradient_penalty(D, lang_features,
                                onesided=opt.onesided_gp, interpolate=(opt.grad_penalty=='wgan'))
                        gp.backward()
                    optimizerD.step()

            # F&C iteration
            utils.unfreeze_net(emb)
            if opt.use_wordemb and opt.fix_emb:
                for lang in emb.langs:
                    emb.wordembs[lang].weight.requires_grad = False
            if opt.use_charemb and opt.fix_charemb:
                emb.charemb.weight.requires_grad = False
            utils.unfreeze_net(F_s)
            map(utils.unfreeze_net, F_p.values())
            utils.unfreeze_net(C)
            utils.freeze_net(D)
            emb.zero_grad()
            if F_s:
                F_s.zero_grad()
            for f_p in F_p.values():
                f_p.zero_grad()
            C.zero_grad()
            # optimizer.zero_grad()
            for lang in opt.langs:
                inputs, targets = utils.endless_get_next_batch(
                        train_loaders, train_iters, lang)
                inputs, lengths, mask, chars, char_lengths = inputs
                bs, seq_len = inputs.size()
                embeds = emb(lang, inputs, chars, char_lengths)
                shared_feat, private_feat = None, None
                if opt.shared_hidden_size > 0:
                    shared_feat = F_s((embeds, lengths))
                if opt.private_hidden_size > 0:
                    private_feat = F_p[lang]((embeds, lengths))
                c_outputs = C((shared_feat, private_feat))
                # targets are padded with -1
                l_c = functional.nll_loss(c_outputs.view(bs*seq_len, -1),
                        targets.view(-1), ignore_index=-1)
                # TODO check retain_graph
                l_c.backward()
                _, pred = torch.max(c_outputs, -1)
                total[lang] += torch.sum(lengths).item()
                correct[lang] += (pred == targets).sum().item()

            # update F with D gradients on all langs
            if D:
                for lang in opt.all_langs:
                    inputs, _ = utils.endless_get_next_batch(
                            unlabeled_loaders, unlabeled_iters, lang)
                    inputs, lengths, mask, chars, char_lengths = inputs
                    embeds = emb(lang, inputs, chars, char_lengths)
                    shared_feat = F_s((embeds, lengths))
                    d_outputs = D((shared_feat, lengths))
                    if opt.loss.lower() == 'gr':
                        d_targets = utils.get_lang_label(opt.loss, lang, len(lengths))
                        l_d = functional.nll_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= -opt.lambd
                    elif opt.loss.lower() == 'bs':
                        d_targets = utils.get_random_lang_label(opt.loss, len(lengths))
                        l_d = functional.kl_div(d_outputs, d_targets, size_average=False)
                        if opt.lambd > 0:
                            l_d *= opt.lambd
                    elif opt.loss.lower() == 'l2':
                        d_targets = utils.get_random_lang_label(opt.loss, len(lengths))
                        l_d = functional.mse_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= opt.lambd
                    l_d.backward()

            optimizer.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        if d_total > 0:
            log.info('D Training Accuracy: {}%'.format(100.0*d_correct/d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.langs))
        log.info('\t'.join([str(100.0*correct[d]/total[d]) for d in opt.langs]))
        log.info('Evaluating validation sets:')
        acc = {}
        for lang in opt.dev_langs:
            acc[lang] = evaluate(f'{lang}_dev', dev_loaders[lang], vocabs[lang], tag_vocab,
                    emb, lang, F_s, F_p[lang] if lang in F_p else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        test_acc = {}
        for lang in opt.dev_langs:
            test_acc[lang] = evaluate(f'{lang}_test', test_loaders[lang], vocabs[lang], tag_vocab,
                    emb, lang, F_s, F_p[lang] if lang in F_p else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average test accuracy: {avg_test_acc}')

        if avg_acc > best_avg_acc:
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_avg_acc = avg_acc
            with open(os.path.join(opt.model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            if F_s:
                torch.save(F_s.state_dict(),
                        '{}/netF_s.pth'.format(opt.model_save_file))
            torch.save(emb.state_dict(),
                    '{}/net_emb.pth'.format(opt.model_save_file))
            for lang in opt.langs:
                if lang in F_p:
                    torch.save(F_p[lang].state_dict(),
                            '{}/net_F_p_{}.pth'.format(opt.model_save_file, lang))
            torch.save(C.state_dict(),
                    '{}/netC.pth'.format(opt.model_save_file))
            if D:
                torch.save(D.state_dict(),
                        '{}/netD.pth'.format(opt.model_save_file))

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')
    return best_acc


def evaluate(name, loader, vocab, tag_vocab, emb, lang, F_s, F_p, C):
    emb.eval()
    if F_s:
        F_s.eval()
    if F_p:
        F_p.eval()
    C.eval()
    it = iter(loader)
    conll = io.StringIO()
    with torch.no_grad():
        for inputs, targets in tqdm(it, ascii=True):
            inputs, lengths, _, chars, char_lengths = inputs
            embeds = (emb(lang, inputs, chars, char_lengths), lengths)
            shared_features, lang_features = None, None
            if opt.shared_hidden_size > 0:
                shared_features = F_s(embeds)
            if opt.private_hidden_size > 0:
                if not F_p:
                    # unlabeled lang
                    lang_features = torch.zeros(targets.size(0),
                            targets.size(1), opt.private_hidden_size).to(opt.device) 
                else:
                    lang_features = F_p(embeds)
            outputs = C((shared_features, lang_features))
            _, pred = torch.max(outputs, -1)
            bs, seq_len = pred.size()
            for i in range(bs):
                for j in range(lengths[i]):
                    word = vocab.get_word(inputs[i][j])
                    gold_tag = tag_vocab.get_tag(targets[i][j])
                    pred_tag = tag_vocab.get_tag(pred[i][j])
                    conll.write(f'{word} {gold_tag} {pred_tag}\n')
                conll.write('\n')
    f1 = utils.conllF1(conll, log)
    if opt.output_pred:
        with open(os.path.join(opt.model_save_file, f"{name}.out"), 'w') as ouf:
            conll.seek(0)
            shutil.copyfileobj(conll, ouf)
    conll.close()
    log.info('{}: F1 score: {}%'.format(name, f1))
    return f1


def main():
    if not os.path.exists(opt.model_save_file):
        os.makedirs(opt.model_save_file)
    log.info('Running the MAN model...')
    vocabs = {}
    tag_vocab = TagVocab()
    assert opt.use_wordemb or opt.use_charemb, "At least one of word or char embeddings must be used!"
    char_vocab = Vocab(opt.charemb_size)
    log.info(f'Loading Datasets...')
    log.info(f'Domain: {opt.domain}')
    log.info(f'Languages {opt.langs}')

    # TODO unlabeled_sets not available yet (always None now)
    log.info('Loading Embeddings...')
    train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
    for lang in opt.all_langs:
        log.info(f'Building Vocab for {lang}...')
        vocabs[lang] = Vocab(opt.emb_size, opt.emb_filenames[lang])
        assert not opt.train_on_translation or not opt.test_on_translation
        if opt.dataset.lower() == 'conll':
            get_dataset_fn = get_conll_ner_datasets
            if opt.train_on_translation:
                get_dataset_fn = get_train_on_translation_conll_ner_datasets
            if opt.test_on_translation:
                get_dataset_fn = get_test_on_translation_conll_ner_datasets
            train_sets[lang], dev_sets[lang], test_sets[lang], unlabeled_sets[lang] = \
                    get_dataset_fn(vocabs[lang], char_vocab, tag_vocab, opt.conll_dir, lang)
        elif opt.dataset.lower() == 'cortana':
            get_dataset_fn = get_cortana_bio_datasets
            if opt.train_on_translation:
                get_dataset_fn = get_train_on_translation_datasets
            if opt.test_on_translation:
                get_dataset_fn = get_test_on_translation_datasets
            train_sets[lang], dev_sets[lang], test_sets[lang], unlabeled_sets[lang] = \
                    get_dataset_fn(vocabs[lang], char_vocab, tag_vocab, opt.cortana_dir, opt.domain, lang)
        else:
            raise Exception(f"Unknown dataset {opt.dataset}")
    opt.num_labels = len(tag_vocab)
    log.info(f'Done Loading Datasets.')

    cv = train(vocabs, char_vocab, tag_vocab, train_sets, dev_sets, test_sets, unlabeled_sets)
    log.info(f'Training done...')
    acc = sum(cv['valid'].values()) / len(cv['valid'])
    log.info(f'Validation Set Domain Average\t{acc}')
    test_acc = sum(cv['test'].values()) / len(cv['test'])
    log.info(f'Test Set Domain Average\t{test_acc}')
    return cv


if __name__ == '__main__':
    main()
