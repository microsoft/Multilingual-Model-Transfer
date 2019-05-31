# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=30)
parser.add_argument('--dataset', default='conll')
parser.add_argument('--amazon_dir', default='./data/amazon/')
parser.add_argument('--conll_dir', default='./data/conll_ner/')
parser.add_argument('--train_on_translation/', dest='train_on_translation', action='store_true')
parser.add_argument('--test_on_translation/', dest='test_on_translation', action='store_true')
# for preprocessed amazon dataset; set to -1 to use 30000
parser.add_argument('--domain', type=str, default='books')
# labeled langs: if not set, will use default langs for the dataset
parser.add_argument('--langs', type=str, nargs='+', default=[])
parser.add_argument('--unlabeled_langs', type=str, nargs='+', default=[])
parser.add_argument('--dev_langs', type=str, nargs='+', default=[])
parser.add_argument('--use_charemb', action='store_true', default=True)
parser.add_argument('--no_charemb', dest='use_charemb', action='store_false')
parser.add_argument('--use_wordemb', dest='use_wordemb', action='store_true', default=True)
parser.add_argument('--no_wordemb', dest='use_wordemb', action='store_false')
# should be (lang, filename) pairs
# will be made available as a dict lang->emb_filename
parser.add_argument('--emb_filenames', type=str, nargs='+', default=[])
# alternatively, choose from a set of default embs (paths hard-coded later in this file)
# muse, muse-idchar, vecmap or mono
parser.add_argument('--default_emb', type=str, default='muse')
parser.add_argument('--emb_size', type=int, default=300)
# char embeddings
parser.add_argument('--charemb_size', type=int, default=128)
parser.add_argument('--charemb_model', type=str, default='cnn') # cnn
parser.add_argument('--charemb_num_layers', type=int, default=1)
parser.add_argument('--fix_charemb', action='store_true', default=True)
parser.add_argument('--no_fix_charemb', action='store_false')
# lowercase the characters (useful for German NER)
parser.add_argument('--lowercase_char', action='store_true', default=False)
# for LSTM D model
parser.add_argument('--charemb_lstm_layers', type=int, default=1)
parser.add_argument('--charemb_attn', default='dot')  # attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--charemb_no_bdrnn', dest='charemb_bdrnn', action='store_false', default=True)  # bi-directional LSTM
# for CNN model
parser.add_argument('--charemb_kernel_num', type=int, default=200)
parser.add_argument('--charemb_kernel_sizes', type=int, nargs='+', default=[3,4,5])

parser.add_argument('--max_seq_len', type=int, default=0) # set to <=0 to not truncate
# if True, training samples with all Os are removed
parser.add_argument('--remove_empty_samples', action='store_true', default=False)
# which data to be used as unlabeled data: train, unlabeled, or both
parser.add_argument('--unlabeled_data', type=str, default='unlabeled')
parser.add_argument('--model_save_file', default='./save/manmoe')
#parser.add_argument('--output_pred', dest='output_pred', action='store_true')
#parser.add_argument('--dump_gate_weights', dest='dump_gate_weights', action='store_true')
parser.add_argument('--test_only', dest='test_only', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--char_batch_size', type=int, default=0)
# In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
# so we will skip leftover batch of size < batch_size
parser.add_argument('--no_skip_leftover_batch', dest='skip_leftover_batch', action='store_false', default=True)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--D_learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--D_weight_decay', type=float, default=1e-8)
# decay lr if validation f1 not increasing
parser.add_argument('--lr_decay', type=float, default=1)
parser.add_argument('--lr_decay_epochs', type=int, default=3)
parser.add_argument('--fix_emb', action='store_true', default=True)
parser.add_argument('--no_fix_emb', action='store_false')
parser.add_argument('--random_emb', action='store_true', default=False)
parser.add_argument('--F_layers', type=int, default=2)
parser.add_argument('--model', default='lstm')
# for LSTM model
parser.add_argument('--bdrnn/', dest='bdrnn', action='store_true', default=True)  # bi-directional LSTM
parser.add_argument('--no_bdrnn/', dest='bdrnn', action='store_false', default=True)  # bi-directional LSTM
# for CNN model
parser.add_argument('--kernel_num', type=int, default=200)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
# D parameters
parser.add_argument('--D_layers', type=int, default=1)
parser.add_argument('--D_model', default='cnn')
# for LSTM D model
parser.add_argument('--D_lstm_layers', type=int, default=1)
parser.add_argument('--D_attn', default='dot')  # attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--D_bdrnn/', dest='D_bdrnn', action='store_true', default=True)  # bi-directional LSTM
parser.add_argument('--D_no_bdrnn/', dest='D_bdrnn', action='store_false', default=True)  # bi-directional LSTM
# for CNN model
parser.add_argument('--D_kernel_num', type=int, default=200)
parser.add_argument('--D_kernel_sizes', type=int, nargs='+', default=[3,4,5])

parser.add_argument('--C_layers', type=int, default=1)
# gate the C input (the features)
parser.add_argument('--C_input_gate/', dest='C_input_gate', action='store_true', default=False)
# see the MAN paper; in this work, we only implement the GR loss
parser.add_argument('--loss', default='gr')
parser.add_argument('--shared_hidden_size', type=int, default=128)
parser.add_argument('--private_hidden_size', type=int, default=128)
# if concat_sp, the shared and private features are concatenated
# otherwise, they are added (and private_hidden_size must be = shared_hidden_size)
parser.add_argument('--concat_sp', dest='concat_sp', action='store_true', default=True)
parser.add_argument('--add_sp', dest='concat_sp', action='store_false')
parser.add_argument('--sp_attn', dest='sp_attn', action='store_true')
parser.add_argument('--sp_sigmoid_attn', dest='sp_sigmoid_attn', action='store_true')
parser.add_argument('--activation', default='relu') # relu, leaky
parser.add_argument('--wgan_trick/', dest='wgan_trick', action='store_true', default=False)
parser.add_argument('--no_wgan_trick/', dest='wgan_trick', action='store_false')
parser.add_argument('--n_critic', type=int, default=2) # hyperparameter k in the paper
parser.add_argument('--lambd', type=float, default=0.002)
# lambda scheduling: not used
parser.add_argument('--lambd_schedule', dest='lambd_schedule', action='store_true', default=False)
# gradient penalty: not used
parser.add_argument('--grad_penalty', dest='grad_penalty', default='none') #none, wgan or dragan
parser.add_argument('--onesided_gp', dest='onesided_gp', action='store_true')
parser.add_argument('--gp_lambd', type=float, default=0.1)
# orthogality penalty: not used
parser.add_argument('--ortho_penalty', dest='ortho_penalty', type=float, default=0)
# batch normalization
parser.add_argument('--F_bn/', dest='F_bn', action='store_true', default=False)
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--C_bn/', dest='C_bn', action='store_true', default=False)
parser.add_argument('--no_C_bn/', dest='C_bn', action='store_false')
parser.add_argument('--D_bn/', dest='D_bn', action='store_true', default=False)
parser.add_argument('--no_D_bn/', dest='D_bn', action='store_false')
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--char_dropout', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--mlp_dropout', type=float, default=0.5)
parser.add_argument('--D_word_dropout', type=float, default=0)
parser.add_argument('--D_dropout', type=float, default=0.5)
parser.add_argument('--device/', dest='device', type=str, default='cuda')
parser.add_argument('--debug/', dest='debug', action='store_true')
###### MoE options ######
# a shared LSTM is used, and each expert is a MLP
parser.add_argument('--F_hidden_size', type=int, default=128)
parser.add_argument('--expert_hidden_size', type=int, default=128)
# if True, a shared expert is added (not used in paper)
parser.add_argument('--expert_sp/', dest='expert_sp', action='store_true', default=False)
parser.add_argument('--gate_loss_weight', type=float, default=0.01)
parser.add_argument('--C_gate_loss_weight', type=float, default=0.01)
parser.add_argument('--moe_last_act', type=str, default='tanh') # tanh or relu
parser.add_argument('--detach_gate_input/', dest='detach_gate_input', action='store_true', default=True)
parser.add_argument('--no_detach_gate_input/', dest='detach_gate_input', action='store_false')
# only effective if MoE is added between LSTMs (otherwise, specify C_layers)
parser.add_argument('--MoE_layers', type=int, default=2)
parser.add_argument('--MoE_bn/', dest='MoE_bn', action='store_true', default=False)
parser.add_argument('--no_MoE_bn/', dest='MoE_bn', action='store_false')
### Cross-Lingual Text Classification Options ###
parser.add_argument('--F_attn', default='dot')  # attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--Fp_MoE', dest='Fp_MoE', action='store_true', default=True)
parser.add_argument('--no_Fp_MoE/', dest='Fp_MoE', action='store_false')
parser.add_argument('--C_MoE', dest='C_MoE', action='store_true', default=True)
parser.add_argument('--no_C_MoE/', dest='C_MoE', action='store_false')
opt = parser.parse_args()

# automatically prepared options
if not torch.cuda.is_available():
    opt.device = 'cpu'

opt.all_langs = opt.langs + opt.unlabeled_langs
if len(opt.dev_langs) == 0:
    opt.dev_langs = opt.all_langs

DEFAULT_EMB = {
    'en-us': './data/embeddings/muse/wiki.200k.en.vec',
    'de-de': './data/embeddings/muse/muse.200k.de-en.de.vec',
    'es-es': './data/embeddings/muse/muse.200k.es-en.es.vec',
    'zh-cn': './data/embeddings/muse/muse.200k.zh-en.zh.vec',
    'en': './data/embeddings/muse/wiki.200k.en.vec',
    'de': './data/embeddings/muse/muse.200k.de-en.de.vec',
    'es': './data/embeddings/muse/muse.200k.es-en.es.vec',
    'zh': './data/embeddings/muse/muse.200k.zh-en.zh.vec',
    'fr': './data/embeddings/muse/muse.200k.fr-en.fr.vec',
    'ja': './data/embeddings/muse/muse.200k.ja-en.ja.vec',
    'eng': './data/embeddings/muse/wiki.200k.en.vec',
    'deu': './data/embeddings/muse/muse.200k.de-en.de.vec',
    'esp': './data/embeddings/muse/muse.200k.es-en.es.vec',
    'ned': './data/embeddings/muse/muse.200k.nl-en.nl.vec'
}
UMWE_EMB = {
    'en': './data/embeddings/wiki.200k.en.vec',
    'de': './data/embeddings/umwe-fullexport/umwe.200k.de-en.de.vec',
    'es': './data/embeddings/umwe-fullexport/umwe.200k.es-en.es.vec',
    'zh': './data/embeddings/umwe-fullexport/umwe.200k.zh-en.zh.vec',
    'fr': './data/embeddings/umwe-fullexport/umwe.200k.fr-en.fr.vec',
    'ja': './data/embeddings/umwe-fullexport/umwe.200k.ja-en.ja.vec',
    'eng': './data/embeddings/wiki.200k.en.vec',
    'deu': './data/embeddings/umwe-fullexport/umwe.200k.deesnl2en.de.vec',
    'esp': './data/embeddings/umwe-fullexport/umwe.200k.deesnl2en.es.vec',
    'ned': './data/embeddings/umwe-fullexport/umwe.200k.deesnl2en.nl.vec'
}
DEFAULT_VECMAP_EMB = {
    'en-us': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.de-en.en.vec',
    'de-de': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.de-en.de.vec',
    'es-es': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.es-en.es.vec',
    'zh-cn': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.zh-en.zh.vec',
    'en': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.de-en.en.vec',
    'de': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.de-en.de.vec',
    'fr': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.fr-en.fr.vec',
    'ja': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.ja-en.ja.vec',
    'eng': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.de-en.en.vec',
    'deu': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.de-en.de.vec',
    'esp': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.es-en.es.vec',
    'ned': './data/embeddings/vecmap/vecmap_idchar_ortho.200k.nl-en.nl.vec',
}

# init emb filenames
assert len(opt.emb_filenames) % 2 == 0, "emb_filenames should be in (lang, filename) pairs"
emb_fns = dict(zip(opt.emb_filenames[::2], opt.emb_filenames[1::2]))
opt.emb_filenames = {}
for i, lang in enumerate(opt.all_langs):
    if opt.default_emb == 'muse':
        def_fn = DEFAULT_EMB[lang]
    elif opt.default_emb == 'vecmap':
        def_fn = DEFAULT_VECMAP_EMB[lang]
    elif opt.default_emb == 'umwe':
        def_fn = UMWE_EMB[lang]
    else:
        def_fn = ''
    fn = emb_fns[lang] if lang in emb_fns else def_fn
    opt.emb_filenames[lang] = fn

opt.max_kernel_size = max(opt.kernel_sizes + opt.D_kernel_sizes)
if opt.activation.lower() == 'relu':
    opt.act_unit = nn.ReLU()
elif opt.activation.lower() == 'leaky':
    opt.act_unit = nn.LeakyReLU()
else:
    raise Exception(f'Unknown activation function {opt.activation}')

opt.total_emb_size = 0
if opt.use_wordemb:
    opt.total_emb_size += opt.emb_size
if opt.use_charemb:
    opt.total_emb_size += opt.charemb_size
