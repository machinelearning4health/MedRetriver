import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
        self.max_pos = max_seq_len

    def forward(self, input_len):

        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([input_len.size(0), self.max_pos])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        return self.position_encoding(input_pos), input_pos


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled


class Attention(nn.Module):
    def __init__(self, in_feature, num_head, dropout):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.fc = nn.Linear(in_feature, in_feature, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = key.size(0)
        res = query
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale
        if attn_mask is not None:
            batch_size, q_len, k_len = attn_mask.size()
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_head, q_len, k_len)
            energy = energy.masked_fill(attn_mask == 0, -np.inf)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x, attention


class EHREncoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super(EHREncoder, self).__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.encoder_layers = nn.ModuleList([Attention(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.positional_feed_forward_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                                           nn.Linear(4 * d_model, d_model))
                                                             for _ in range(num_layers)])
        self.pooler = MaxPoolLayer()
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))

    def forward(self, input_seqs, masks, lengths):
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        output_pos, ind_pos = self.pos_emb(lengths)
        x += output_pos
        attentions = []
        outputs = []
        for i in range(len(self.encoder_layers)):
            x, attention = self.encoder_layers[i](x, x, x, masks)
            res = x
            x = self.positional_feed_forward_layers[i](x)
            x = self.dropout(x)
            x = self.layer_norm(x + res)
            attentions.append(attention)
            outputs.append(x)

        return outputs[-1]


class HitaNet(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super(HitaNet, self).__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(d_model))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)
        self.encoder_layers = nn.ModuleList([Attention(d_model, num_heads, dropout) for _ in range(1)])
        self.positional_feed_forward_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                                           nn.Linear(4 * d_model, d_model))
                                                             for _ in range(1)])
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.selection_time_layer = nn.Linear(1, 64)
        self.weight_layer = torch.nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.self_layer = torch.nn.Linear(256, 1)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))

    def forward(self, input_seqs, masks, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        x = self.embbedding(input_seqs).sum(dim=2) + self.bias_embedding
        x = self.emb_dropout(x)
        bs, seq_length, d_model = x.size()
        output_pos, ind_pos = self.pos_emb(lengths)
        x += output_pos
        x += time_feature
        attentions = []
        outputs = []
        for i in range(len(self.encoder_layers)):
            x, attention = self.encoder_layers[i](x, x, x, masks)
            res = x
            x = self.positional_feed_forward_layers[i](x)
            x = self.dropout(x)
            x = self.layer_norm(x + res)
            attentions.append(attention)
            outputs.append(x)
        final_statues = outputs[-1].gather(1, lengths[:, None, None].expand(bs, 1, d_model) - 1).expand(bs, seq_length,
                                                                                                        d_model)
        quiryes = self.relu(self.quiry_layer(final_statues))
        mask = (torch.arange(seq_length, device=x.device).unsqueeze(0).expand(bs, seq_length) >= lengths.unsqueeze(1))
        self_weight = torch.softmax(self.self_layer(outputs[-1]).squeeze().masked_fill(mask, -np.inf), dim=1).view(bs,
                                                                                                                   seq_length).unsqueeze(
            2)
        selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step)))
        selection_feature = torch.sum(selection_feature * quiryes, 2) / 8
        time_weight = torch.softmax(selection_feature.masked_fill(mask, -np.inf), dim=1).view(bs, seq_length).unsqueeze(
            2)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2).view(bs, seq_length, 2)
        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = outputs[-1] * total_weight.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        prediction = self.output_mlp(averaged_features)
        return averaged_features, outputs[-1]


class LSAN(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder_layers = nn.ModuleList([Attention(d_model, num_heads, dropout) for _ in range(1)])
        self.positional_feed_forward_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                                           nn.Linear(4 * d_model, d_model))
                                                             for _ in range(1)])
        self.pooler = MaxPoolLayer()
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.MATT = nn.Sequential(nn.Linear(d_model, int(d_model / 4), bias=False),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(int(d_model / 4), int(d_model / 8), bias=False),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(int(d_model / 8), 1))
        visit_ATT_dim = 2 * d_model
        self.visit_ATT = nn.Sequential(nn.Linear(visit_ATT_dim, int(visit_ATT_dim / 4)),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(visit_ATT_dim / 4), int(visit_ATT_dim / 8)),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(visit_ATT_dim / 8), 1))
        self.Classifier = nn.Linear(2 * d_model, 2)
        self.local_conv_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3,
                                          padding=1)

    def forward(self, input_seqs, masks, lengths):
        input_embedding = self.embbedding(input_seqs)
        bs, seqlen, numcode, d_model = input_embedding.size()
        input_embedding = input_embedding.view(bs * seqlen, numcode, d_model)
        attn_weight = F.softmax(self.MATT(input_embedding), dim=1)
        diag_result_att = torch.matmul(attn_weight.permute(0, 2, 1), input_embedding).squeeze(1)
        diag_result_att = diag_result_att.view(bs, seqlen, d_model)
        diag_result_att = self.emb_dropout(diag_result_att)
        output_pos, ind_pos = self.pos_emb(lengths)
        x = diag_result_att + output_pos
        attentions = []
        outputs = []
        for i in range(len(self.encoder_layers)):
            x, attention = self.encoder_layers[i](x, x, x, masks)
            res = x
            x = self.positional_feed_forward_layers[i](x)
            x = self.dropout(x)
            x = self.layer_norm(x + res)
            attentions.append(attention)
            outputs.append(x)

        local_conv_feat = self.local_conv_layer(diag_result_att.permute(0, 2, 1))
        concat_feat = torch.cat((outputs[-1], local_conv_feat.permute(0, 2, 1)), dim=2)
        visit_attn_weight = torch.softmax(self.visit_ATT(concat_feat), dim=1)
        visit_result_att = torch.matmul(visit_attn_weight.permute(0, 2, 1), concat_feat).squeeze(1)
        prediction_output = self.Classifier(visit_result_att)
        return prediction_output


class LSTM_encoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, input_seqs, lengths):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        rnn_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)

        return x


class GRUSelf(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.weight_layer = nn.Linear(d_model, 1)

    def forward(self, input_seqs, lengths):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        rnn_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.gru(rnn_input)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        weight = self.weight_layer(rnn_output)
        mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        att = torch.softmax(weight.squeeze().masked_fill(mask, -np.inf), dim=1).view(batch_size, seq_len)
        weighted_features = rnn_output * att.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        pred = self.output_mlp(averaged_features)
        return averaged_features, rnn_output


class Timeline(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.hidden_dim = d_model
        self.word_embeddings = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=False)
        self.hidden2label = nn.Linear(d_model, 2)
        self.attention = nn.Linear(d_model, d_model)
        self.vector1 = nn.Parameter(torch.randn(d_model, 1))
        self.decay = nn.Parameter(torch.FloatTensor([-0.1] * (vocab_size + 1)))
        self.initial = nn.Parameter(torch.FloatTensor([1.0] * (vocab_size + 1)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.attention_dimensionality = d_model
        self.WQ1 = nn.Linear(d_model, d_model, bias=False)
        self.WK1 = nn.Linear(d_model, d_model, bias=False)
        self.embed_drop = nn.Dropout(p=dropout_emb)

    def forward(self, input_seqs, masks, lengths, seq_time_step):
        sentence = input_seqs, masks, seq_time_step
        numcode = sentence[0].size()[2]
        numvisit = sentence[0].size()[1]
        numbatch = sentence[0].size()[0]
        thisembeddings = self.word_embeddings(sentence[0].view(-1, numcode))
        thisembeddings = self.embed_drop(thisembeddings)
        myQ1 = self.WQ1(thisembeddings)
        myK1 = self.WK1(thisembeddings)
        dproduct1 = torch.bmm(myQ1, torch.transpose(myK1, 1, 2)).view(numbatch, numvisit, numcode, numcode)
        dproduct1 = dproduct1 - sentence[1].view(numbatch, numvisit, 1, numcode) - sentence[1].view(numbatch, numvisit,
                                                                                                    numcode, 1)
        sproduct1 = self.softmax(dproduct1.view(-1, numcode) / np.sqrt(self.attention_dimensionality)).view(-1, numcode,
                                                                                                            numcode)
        fembedding11 = torch.bmm(sproduct1, thisembeddings)
        fembedding11 = (((sentence[1] - (1e+20)) / (-1e+20)).view(-1, numcode, 1) * fembedding11)
        mydecay = self.decay[sentence[0].view(-1)].view(numvisit * numbatch, numcode, 1)
        myini = self.initial[sentence[0].view(-1)].view(numvisit * numbatch, numcode, 1)
        temp1 = torch.bmm(mydecay, sentence[2].view(-1, 1, 1))
        temp2 = self.sigmoid(temp1 + myini)
        vv = torch.bmm(temp2.view(-1, 1, numcode), fembedding11)
        vv = vv.view(numbatch, numvisit, -1).transpose(0, 1)
        lstm_out, hidden = self.lstm(vv)
        mask_final = torch.arange(input_seqs.size(1), device=input_seqs.device).unsqueeze(0).expand(input_seqs.size(0),
                                                                                                    input_seqs.size(
                                                                                                        1)) == lengths.unsqueeze(
            1) - 1
        lstm_out_final = lstm_out * mask_final.float().transpose(0, 1).view(numvisit, numbatch, 1)
        lstm_out_final = lstm_out_final.sum(dim=0)
        label_space = self.hidden2label(lstm_out_final)
        return lstm_out.transpose(0, 1)


class SAND(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.encoder_layer = Attention(d_model, num_heads, dropout)
        self.positional_feed_forward_layer = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                           nn.Linear(4 * d_model, d_model))
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(d_model))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)
        self.drop_out = nn.Dropout(dropout)
        self.out_layer = nn.Linear(d_model * 4, 2)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_seqs, masks, lengths):
        x = self.embbedding(input_seqs).sum(dim=2) + self.bias_embedding
        bs, sl, dm = x.size()
        output_pos, ind_pos = self.pos_emb(lengths)
        x += output_pos
        x, attention = self.encoder_layer(x, x, x, masks)

        mask = (torch.arange(sl, device=x.device).unsqueeze(0).expand(bs, sl) >= lengths.unsqueeze(
            1))
        x = x.masked_fill(mask.unsqueeze(-1).expand_as(x), 0.0)
        U = torch.zeros((x.size(0), 4, x.size(2))).to(x.device)
        lengths = lengths.float()
        for t in range(1, input_seqs.size(1) + 1):
            s = 4 * t / lengths
            for m in range(1, 4 + 1):
                w = torch.pow(1 - torch.abs(s - m) / 4, 2)
                U[:, m - 1] += w.unsqueeze(-1) * x[:, t - 1]
        U = U.view(input_seqs.size(0), -1)
        U = self.drop_out(U)
        output = self.out_layer(U)
        return U, x


class Retain(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.variable_level_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.visit_level_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.variable_level_attention = nn.Linear(d_model, d_model)
        self.visit_level_attention = nn.Linear(d_model, 1)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 2)

        self.var_hidden_size = d_model

        self.visit_hidden_size = d_model

    def forward(self, input_seqs):
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.dropout(x)
        visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(x)
        alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = torch.softmax(alpha, dim=1)
        var_rnn_output, var_rnn_hidden = self.variable_level_rnn(x)
        beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)
        attn_w = visit_attn_w * var_attn_w
        c_all = attn_w * x
        c = torch.sum(c_all, dim=1)
        c = self.output_dropout(c)
        output = self.output_layer(c)
        return c, c_all


class RetainEx(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding1 = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.embbedding2 = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.RNN1 = nn.LSTM(d_model + 3, d_model,
                            1, batch_first=True)
        self.RNN2 = nn.LSTM(d_model + 3, d_model,
                            1, batch_first=True)
        self.wa = nn.Linear(d_model, 1, bias=False)
        self.Wb = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, 2, bias=False)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, input_seqs, time_step):
        embedded = self.embbedding1(input_seqs).sum(dim=2)
        embedded2 = self.embbedding2(input_seqs).sum(dim=2)
        b, seq, features = embedded.size()
        dates = torch.stack([time_step, 1 / (time_step + 1), 1 / torch.log(np.e + time_step)], 2)
        embedded = torch.cat([embedded, dates], 2)
        outputs1 = self.RNN1(embedded)[0]
        outputs2 = self.RNN2(embedded)[0]
        E = self.wa(outputs1.contiguous().view(b * seq, -1))
        alpha = F.softmax(E.view(b, seq), 1)
        outputs2 = self.Wb(outputs2.contiguous().view(b * seq, -1))
        Beta = torch.tanh(outputs2).view(b, seq, features)
        v_all = (embedded2 * Beta) * alpha.unsqueeze(2).expand(b, seq, features)
        outputs = v_all.sum(1)
        outputs = self.drop_out(outputs)
        return outputs, v_all


def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    ancestors = np.array(list(treeMap.values())).astype('int32')
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    return leaves, ancestors


class Gram(nn.Module):
    def __init__(self, inputDimSize, numAncestors, d_model, dropout, num_layers, treeFile, device):
        super().__init__()
        self.inputDimSize = inputDimSize
        self.W_emb = nn.Embedding(inputDimSize + numAncestors, d_model)
        self.att_MLP = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.softmax = nn.Softmax(1)
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(d_model, 2)
        self.input_dropout = nn.Dropout(0.1)
        leavesList = []
        ancestorsList = []
        for i in range(5, 0, -1):
            leaves, ancestors = build_tree(treeFile + '.level' + str(i) + '.pk')
            sharedLeaves = torch.LongTensor(leaves).to(device)
            sharedAncestors = torch.LongTensor(ancestors).to(device)
            leavesList.append(sharedLeaves)
            ancestorsList.append(sharedAncestors)
        self.leavesList = leavesList
        self.ancestorsList = ancestorsList

    def forward(self, x, lengths):
        embList = []
        for leaves, ancestors in zip(self.leavesList, self.ancestorsList):
            attentionInput = torch.cat((self.W_emb(leaves), self.W_emb(ancestors)), dim=2)
            preAttention = self.att_MLP(attentionInput)
            attention = self.softmax(preAttention)
            tempEmb = self.W_emb(ancestors) * attention
            tempEmb = torch.sum(tempEmb, dim=1)
            embList.append(tempEmb)
        emb = torch.cat(embList, dim=0)
        pad_emb = emb.new_zeros((1, emb.size(1)))
        emb = torch.cat((emb, pad_emb), dim=0)
        assert (lengths > 0).all()
        assert emb.size(0) == self.inputDimSize + 1
        bz, seq_len, num_per_visit = x.size()
        x = emb[x]
        x = self.input_dropout(x)
        x = torch.sum(x, dim=2)
        rnn_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        x = self.pooler(x, lengths)
        x = self.output_mlp(x)
        return x


if __name__ == '__main__':
    output = np.triu(np.ones((3, 3)), k=1)
    print('output: ', output)
