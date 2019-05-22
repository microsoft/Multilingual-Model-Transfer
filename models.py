import torch
import torch.nn.functional as functional
from torch import autograd, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers import *
from options import opt
import utils


class MultiLangWordEmb(nn.Module):
    def __init__(self, vocabs, char_vocab, use_wordemb, use_charemb):
        super(MultiLangWordEmb, self).__init__()
        self.langs = list(vocabs.keys())
        self.use_wordemb = use_wordemb
        self.use_charemb = use_charemb
        if use_wordemb:
            self.wordembs = {}
            for lang, vocab in vocabs.items():
                self.wordembs[lang] = vocab.init_embed_layer(opt.fix_emb)
                self.add_module(f'wordemb_{lang}', self.wordembs[lang])
        if use_charemb:
            self.charemb = char_vocab.init_embed_layer(opt.fix_charemb)
            if opt.charemb_model.lower() == 'lstm':
                charemb_args = {
                    'num_layers': opt.charemb_lstm_layers,
                    'input_size': char_vocab.emb_size,
                    'hidden_size': opt.charemb_size,
                    'word_dropout': opt.char_dropout,
                    'dropout': opt.dropout,
                    'bdrnn': opt.charemb_bdrnn,
                    'attn_type': opt.charemb_attn
                }
            elif opt.charemb_model.lower() == 'cnn':
                charemb_args = {
                    'num_layers': 1,
                    'input_size': char_vocab.emb_size,
                    'hidden_size': opt.charemb_size,
                    'kernel_num': opt.charemb_kernel_num,
                    'kernel_sizes': opt.charemb_kernel_sizes,
                    'word_dropout': opt.char_dropout,
                    'dropout': opt.dropout
                }
            else:
                raise Exception("Unknown Char Embedding Model")
            if opt.charemb_model.lower() == 'lstm':
                self.char_pool = LSTMSequencePooling(**charemb_args)
            elif opt.charemb_model.lower() == 'cnn':
                self.char_pool = CNNSequencePooling(**charemb_args)

    def get_char_vec(self, char_emb, char_lengths):
        bs = opt.char_batch_size
        if bs <= 0:
            return self.char_pool((char_emb, char_lengths))
        # devide into sub-batches if necessary
        char_vec = torch.empty(len(char_emb), opt.charemb_size,
                requires_grad=char_emb.requires_grad).to(char_emb.device)
        for k in range(0, len(char_emb), bs):
            x = char_emb[k:k+bs]
            l = char_lengths[k:k+bs]
            char_vec[k:k+bs] = self.char_pool((x, l))
        return char_vec

    def forward(self, lang, words, chars, char_lengths):
        word_vec, char_vec = None, None
        if self.use_wordemb:
            word_vec = self.wordembs[lang](words)
        if self.use_charemb:
            bs, sl, cl = chars.size()
            char_vec = self.charemb(chars).view(bs*sl, cl, -1)
            #char_vec = self.char_pool((char_vec, char_lengths.view(bs*sl))).view(bs, sl, -1)
            char_vec = self.get_char_vec(char_vec, char_lengths.view(bs*sl)).view(bs, sl, -1)
        if not self.use_wordemb:
            embeds = char_vec
        elif not self.use_charemb:
            embeds = word_vec
        else:
            embeds = torch.cat([word_vec, char_vec], -1)
        return embeds


class CNNSequencePooling(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 word_dropout,
                 dropout):
        super(CNNSequencePooling, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.word_dropout = word_dropout
        self.dropout = dropout

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, input_size)) for K in kernel_sizes])
        
        # at least 1 hidden layer so that the output size is hidden_size
        assert num_layers > 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                        nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            # if batch_norm:
            #     self.fcnet.add_module('f-bn-{}'.format(i), nn.LayerNorm(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), opt.act_unit)

    def forward(self, input):
        data, _ = input
        # conv
        if self.word_dropout > 0:
            data = functional.dropout(data, self.word_dropout, self.training)
        data = data.unsqueeze(1) # batch_size, 1, seq_len, input_size
        x = [functional.relu(conv(data)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)


class LSTMSequencePooling(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 word_dropout,
                 dropout,
                 bdrnn,
                 attn_type):
        super(LSTMSequencePooling, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.bdrnn = bdrnn
        self.attn_type = attn_type
        self.input_size = input_size
        self.hidden_size = hidden_size//2 if bdrnn else hidden_size
        self.n_cells = self.num_layers*2 if bdrnn else self.num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)
        if attn_type == 'dot':
            self.attn = DotAttentionLayer(hidden_size)

    def forward(self, input):
        data, lengths = input
        # lengths_list = lengths.tolist()
        batch_size = len(data)
        if self.word_dropout > 0:
            data = functional.dropout(data, self.word_dropout, self.training)
        packed = pack_padded_sequence(data, lengths, batch_first=True)
        output, (ht, ct) = self.rnn(packed)

        if self.attn_type == 'last':
            return ht[-1] if not self.bdrnn \
                          else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        elif self.attn_type == 'avg':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return torch.sum(unpacked_output, 1) / lengths.float().view(-1, 1)
        elif self.attn_type == 'dot':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return self.attn((unpacked_output, lengths))
        else:
            raise Exception('Please specify valid attention (pooling) mechanism')


class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 emb_size,
                 num_layers,
                 hidden_size,
                 word_dropout,
                 dropout,
                 bdrnn):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.bdrnn = bdrnn
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.hidden_size = hidden_size//2 if bdrnn else hidden_size
        # self.n_cells = self.num_layers*2 if bdrnn else self.num_layers
        
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=self.hidden_size,
                num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)

    def forward(self, input):
        embeds, lengths = input
        # lengths_list = lengths.tolist()
        batch_size, seq_len = embeds.size(0), embeds.size(1)
        # word_dropout before LSTM
        if self.word_dropout > 0:
            embeds = functional.dropout(embeds, self.word_dropout, self.training)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        # state_shape = self.n_cells, batch_size, self.hidden_size
        # h0 = c0 = autograd.Variable(embeds.data.new(*state_shape))
        output, _ = self.rnn(packed, None)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=seq_len)
        return output


class SpMlpTagger(nn.Module):
    def __init__(self,
                 num_layers,
                 shared_input_size,
                 private_input_size,
                 concat_sp,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SpMlpTagger, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.concat_sp = concat_sp
        self.shared_input_size = shared_input_size
        self.private_input_size = private_input_size
        self.input_size = shared_input_size + private_input_size if concat_sp else shared_input_size
        self.hidden_size = hidden_size
        if opt.sp_attn:
            if concat_sp:
                self.sp_attn = nn.Linear(self.input_size, 2)
            else:
                self.sp_attn = nn.Linear(self.input_size, 1)
        if opt.C_input_gate:
            self.input_gate = nn.Linear(self.input_size, self.input_size)
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(self.input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.LayerNorm(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        fs, fp = input
        if self.shared_input_size == 0:
            features = fp
        elif self.private_input_size == 0:
            features = fs
        else:
            if self.concat_sp:
                if opt.sp_attn:
                    features = torch.cat([fs, fp], dim=-1)
                    # bs x seqlen x 2
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(self.sp_attn(features))
                    else:
                        alphas = functional.softmax(self.sp_attn(features), dim=-1)
                    fs = fs * alphas[:,:,0].unsqueeze(-1)
                    fp = fp * alphas[:,:,1].unsqueeze(-1)
                    features = torch.cat([fs, fp], dim=-1)
                else:
                    features = torch.cat([fs, fp], dim=-1)
            else:
                if opt.sp_attn:
                    a1 = self.sp_attn(fs)
                    a2 = self.sp_attn(fp)
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(torch.cat([a1, a2], dim=-1))
                    else:
                        alphas = functional.softmax(torch.cat([a1, a2], dim=-1), dim=-1)
                    features = torch.stack([fs, fp], dim=2) # bs x seq_len x 2 x hidden_dim
                    features = torch.sum(alphas.unsqueeze(-1) * features, dim=2)
                else:
                    features = fs + fp
        if opt.C_input_gate:
            gates = self.input_gate(features.detach())
            gates = functional.sigmoid(gates)
            features = gates * features
        return self.net(features) # * mask.unsqueeze(2)


class SpClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 shared_input_size,
                 private_input_size,
                 concat_sp,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SpClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.concat_sp = concat_sp
        self.shared_input_size = shared_input_size
        self.private_input_size = private_input_size
        self.input_size = shared_input_size + private_input_size if concat_sp else shared_input_size
        self.hidden_size = hidden_size
        if opt.sp_attn:
            if concat_sp:
                self.sp_attn = nn.Linear(self.input_size, 2)
            else:
                self.sp_attn = nn.Linear(self.input_size, 1)
        if opt.C_input_gate:
            self.input_gate = nn.Linear(self.input_size, self.input_size)
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(self.input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.LayerNorm(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        fs, fp = input
        if self.shared_input_size == 0:
            features = fp
        elif self.private_input_size == 0:
            features = fs
        else:
            if self.concat_sp:
                if opt.sp_attn:
                    features = torch.cat([fs, fp], dim=-1)
                    # bs x 2
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(self.sp_attn(features))
                    else:
                        alphas = functional.softmax(self.sp_attn(features), dim=-1)
                    fs = fs * alphas[:,0].unsqueeze(-1)
                    fp = fp * alphas[:,1].unsqueeze(-1)
                    features = torch.cat([fs, fp], dim=-1)
                else:
                    features = torch.cat([fs, fp], dim=-1)
            else:
                if opt.sp_attn:
                    a1 = self.sp_attn(fs)
                    a2 = self.sp_attn(fp)
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(torch.cat([a1, a2], dim=-1))
                    else:
                        alphas = functional.softmax(torch.cat([a1, a2], dim=-1), dim=-1)
                    features = torch.stack([fs, fp], dim=2) # bs x 2 x hidden_dim
                    features = torch.sum(alphas.unsqueeze(-1) * features, dim=-2)
                else:
                    features = fs + fp
        if opt.C_input_gate:
            gates = self.input_gate(features.detach())
            gates = functional.sigmoid(gates)
            features = gates * features
        return self.net(features) # * mask.unsqueeze(2)


class Mlp(nn.Module):
    """
    Use tanh for the last layer to better work like LSTM output
    """
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(Mlp, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
                hsize = hidden_size
            elif i+1 == num_layers:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, output_size))
                hsize = output_size
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.LayerNorm(hsize))
            if i+1 < num_layers:
                self.net.add_module('p-relu-{}'.format(i), opt.act_unit)
            else:
                if opt.moe_last_act == 'tanh':
                    self.net.add_module('p-tanh-{}'.format(i), nn.Tanh())
                elif opt.moe_last_act == 'relu':
                    self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
                else:
                    raise NotImplemented(opt.moe_last_act)

    def forward(self, input):
        return self.net(input)


class MlpTagger(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(MlpTagger, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.LayerNorm(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class MixtureOfExperts(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 num_experts,
                 hidden_size,
                 output_size,
                 dropout,
                 bn=False,
                 is_tagger=True):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.gates = nn.Linear(input_size, num_experts)
        mlp = MlpTagger if is_tagger else Mlp
        self.experts = nn.ModuleList([mlp(num_layers, \
                input_size, hidden_size, output_size, dropout, bn) \
                for _ in range(num_experts)])

    def forward(self, input):
        # input: bs x seqlen x input_size
        gate_input = input.detach() if opt.detach_gate_input else input
        gate_outs = self.gates(gate_input)
        gate_softmax = functional.softmax(gate_outs, dim=-1) # bs x seqlen x #experts
        # bs x seqlen x #experts x output_size
        expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
        # bs x seqlen x output_size
        output = torch.sum(gate_softmax.unsqueeze(-1) * expert_outs, dim=-2)
        # output logits
        return output, gate_outs


class SpMixtureOfExperts(nn.Module):
    def __init__(self,
                 num_layers,
                 shared_input_size,
                 private_input_size,
                 concat_sp,
                 num_experts,
                 hidden_size,
                 output_size,
                 dropout,
                 bn):
        super(SpMixtureOfExperts, self).__init__()
        self.shared_input_size = shared_input_size
        self.private_input_size = private_input_size
        self.concat_sp = concat_sp
        input_size = shared_input_size + private_input_size if concat_sp else shared_input_size
        if opt.sp_attn:
            if concat_sp:
                self.sp_attn = nn.Linear(input_size, 2)
            else:
                self.sp_attn = nn.Linear(input_size, 1)
        self.moe = MixtureOfExperts(num_layers, input_size, num_experts,
                hidden_size, output_size, dropout, bn, is_tagger=True)

    def forward(self, input):
        fs, fp = input
        if self.shared_input_size == 0:
            features = fp
        elif self.private_input_size == 0:
            features = fs
        else:
            if self.concat_sp:
                if opt.sp_attn:
                    features = torch.cat([fs, fp], dim=-1)
                    # bs x seqlen x 2
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(self.sp_attn(features))
                    else:
                        alphas = functional.softmax(self.sp_attn(features), dim=-1)
                    fs = fs * alphas[:,:,0].unsqueeze(-1)
                    fp = fp * alphas[:,:,1].unsqueeze(-1)
                    features = torch.cat([fs, fp], dim=-1)
                else:
                    features = torch.cat([fs, fp], dim=-1)
            else:
                if opt.sp_attn:
                    a1 = self.sp_attn(fs)
                    a2 = self.sp_attn(fp)
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(torch.cat([a1, a2], dim=-1))
                    else:
                        alphas = functional.softmax(torch.cat([a1, a2], dim=-1), dim=-1)
                    features = torch.stack([fs, fp], dim=2) # bs x seq_len x 2 x hidden_dim
                    features = torch.sum(alphas.unsqueeze(-1) * features, dim=2)
                else:
                    features = fs + fp
        return self.moe(features)


class SpAttnMixtureOfExperts(nn.Module):
    def __init__(self,
                 num_layers,
                 shared_input_size,
                 private_input_size,
                 concat_sp,
                 num_experts,
                 hidden_size,
                 output_size,
                 dropout,
                 attn_type,
                 bn):
        super(SpAttnMixtureOfExperts, self).__init__()
        self.shared_input_size = shared_input_size
        self.private_input_size = private_input_size
        self.concat_sp = concat_sp
        input_size = shared_input_size + private_input_size if concat_sp else shared_input_size
        if opt.sp_attn:
            if concat_sp:
                self.sp_attn = nn.Linear(input_size, 2)
            else:
                self.sp_attn = nn.Linear(input_size, 1)
        self.moe = MixtureOfExperts(num_layers, input_size, num_experts,
                hidden_size, output_size, dropout, bn, is_tagger=True)
        if attn_type == 'dot':
            self.attn = DotAttentionLayer(hidden_size)

    def forward(self, input):
        fs, fp, lengths = input
        if self.shared_input_size == 0:
            features = fp
        elif self.private_input_size == 0:
            features = fs
        else:
            if self.concat_sp:
                if opt.sp_attn:
                    features = torch.cat([fs, fp], dim=-1)
                    # bs x seqlen x 2
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(self.sp_attn(features))
                    else:
                        alphas = functional.softmax(self.sp_attn(features), dim=-1)
                    fs = fs * alphas[:,:,0].unsqueeze(-1)
                    fp = fp * alphas[:,:,1].unsqueeze(-1)
                    features = torch.cat([fs, fp], dim=-1)
                else:
                    features = torch.cat([fs, fp], dim=-1)
            else:
                if opt.sp_attn:
                    a1 = self.sp_attn(fs)
                    a2 = self.sp_attn(fp)
                    if opt.sp_sigmoid_attn:
                        alphas = functional.sigmoid(torch.cat([a1, a2], dim=-1))
                    else:
                        alphas = functional.softmax(torch.cat([a1, a2], dim=-1), dim=-1)
                    features = torch.stack([fs, fp], dim=2) # bs x seq_len x 2 x hidden_dim
                    features = torch.sum(alphas.unsqueeze(-1) * features, dim=2)
                else:
                    features = fs + fp
        features = self.attn((features, lengths))
        return self.moe(features)


class MLPLanguageDiscriminator(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 num_langs,
                 loss_type,
                 dropout,
                 batch_norm=False):
        super(MLPLanguageDiscriminator, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.num_langs = num_langs
        self.loss_type = loss_type
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('d-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('d-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('d-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('d-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('d-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('d-linear-final', nn.Linear(hidden_size, num_langs))
        if loss_type.lower() == 'gr' or loss_type.lower() == 'bs':
            self.net.add_module('d-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        scores = self.net(input)
        if self.loss_type.lower() == 'l2':
            # normalize
            scores = functional.relu(scores)
            scores /= torch.sum(scores, dim=1, keepdim=True)
        return scores


class LanguageDiscriminator(nn.Module):
    def __init__(self,
                 model,
                 mlp_layers,
                 input_size,
                 hidden_size,
                 num_langs,
                 dropout,
                 batch_norm=False,
                 model_args=None):
        """
        If model is CNN/LSTM, D takes a sequence and predicts a scalar (model_args dictates architecture)
        If model is MLP, D only takes a single token (and D should be applied individually to all tokens
        If num_langs > 1, D uses a MAN-style model
        If num_langs == 1, D discriminates real vs fake for a specific language
        """
        super(LanguageDiscriminator, self).__init__()
        self.input_size = input_size
        self.model = model
        self.num_langs = num_langs
        if self.model.lower() == 'lstm':
            self.pool = LSTMSequencePooling(**model_args)
        elif self.model.lower() == 'cnn':
            self.pool = CNNSequencePooling(**model_args)
        elif self.model.lower() == 'mlp':
            self.pool = None
        else:
            raise Exception('Please specify valid model architecture')

        self.net = nn.Sequential()
        for i in range(mlp_layers):
            if dropout > 0:
                self.net.add_module('d-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('d-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('d-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('d-bn-{}'.format(i), nn.LayerNorm(hidden_size))
            self.net.add_module('d-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('d-linear-final', nn.Linear(hidden_size, num_langs))
        if opt.loss.lower() == 'gr' or opt.loss.lower() == 'bs':
            if num_langs > 1:
                self.net.add_module('d-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        data, lengths = input
        if self.pool:
            data = self.pool((data, lengths))
        # data: bs x input_size (with pooling)
        # bs*seq_len x input_size  (without pooling)
        scores = self.net(data)
        if opt.loss.lower() == 'l2':
            # normalize
            scores = functional.relu(scores)
            scores /= torch.sum(scores, dim=1, keepdim=True)
        return scores
