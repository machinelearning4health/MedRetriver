import math
import numpy as np
import torch
import torch.nn as nn
from models.modeling_encoder import *


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


class MemoryUpdate(nn.Module):
    def __init__(self, d_model, dropout, blk_emb, blk_trans, mem_size):
        super(MemoryUpdate, self).__init__()
        self.blk_emb = blk_emb
        self.blk_trans = blk_trans
        self.linear_q = nn.Sequential(nn.Linear(d_model, d_model, bias=False))
        self.linear_k = nn.Sequential(nn.Linear(d_model, d_model, bias=False))
        self.linear_v = nn.Sequential(nn.Linear(d_model, d_model, bias=False))

        self.mem_size = mem_size

    def forward(self, mem_input, mask, query):
        input_store = mem_input
        assert mem_input.size() == mask.size()
        mem_input = self.blk_trans(self.blk_emb(mem_input))
        bs, mem_len, d_model = mem_input.size()
        query = self.linear_q(query)
        key = self.linear_k(mem_input)
        value = self.linear_v(mem_input)
        scale = np.sqrt(query.size(-1))
        energy = torch.matmul(query.unsqueeze(1), key.permute(0, 2, 1)) / scale
        attention = torch.softmax(energy, dim=-1)
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(1) == 0, 0)
        attention_topk, indices = torch.topk(attention, self.mem_size, dim=-1)
        mem_output = input_store.gather(1, indices.squeeze())
        if mask is not None:
            mask = mask.gather(1, indices.squeeze())
        return mem_output, mask


class TextDecoder_hita(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_hita, self).__init__()
        self.ehr_encoder = HitaNet(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        v_final, v_all = self.ehr_encoder(input_seqs, masks_ehr, lengths, time_step)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths) + v_final
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, memory_final


class TextDecoder_lsan(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_lsan, self).__init__()
        self.ehr_encoder = LSAN(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        v_e, v_all = self.ehr_encoder(input_seqs, masks_ehr, lengths)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1) + v_e)
        return output, memory_final


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder, self).__init__()
        self.ehr_encoder = EHREncoder(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        v_all = self.ehr_encoder(input_seqs, masks_ehr, lengths)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, memory_final


class TextDecoder_lstm(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_lstm, self).__init__()
        self.ehr_encoder = LSTM_encoder(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        v_all = self.ehr_encoder(input_seqs, lengths)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, torch.stack(memory[1:], dim=1)


class TextDecoder_gruself(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_gruself, self).__init__()
        self.ehr_encoder = GRUSelf(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(3 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        encoder_vec, v_all = self.ehr_encoder(input_seqs, lengths)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final, encoder_vec), dim=1))
        return output, memory_final


class TextDecoder_timeline(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_timeline, self).__init__()
        self.ehr_encoder = Timeline(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        v_all = self.ehr_encoder(input_seqs, code_mask, lengths, time_step)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, memory_final


class TextDecoder_sand(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_sand, self).__init__()
        self.ehr_encoder = SAND(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(6 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        encoder_vec, v_all = self.ehr_encoder(input_seqs, masks_ehr, lengths)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final, encoder_vec), dim=1))
        return output, memory


class TextDecoder_retain(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_retain, self).__init__()
        self.ehr_encoder = Retain(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        encoder_vec, v_all = self.ehr_encoder(input_seqs)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, memory_final


class TextDecoder_retainEx(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_retainEx, self).__init__()
        self.ehr_encoder = RetainEx(vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        encoder_vec, v_all = self.ehr_encoder(input_seqs, time_step)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, torch.stack(memory[1:], dim=1)


class TextDecoder_gram(nn.Module):
    def __init__(self, vocab_size, numAncestors, treeFile, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos,
                 blk_emb_fpath, target_disease_path, target_att_heads, mem_size, mem_update_size, device):
        super(TextDecoder_gram, self).__init__()
        self.ehr_encoder = Gram(vocab_size, numAncestors, d_model, dropout, num_layers, treeFile, device)
        emb = np.load(blk_emb_fpath)
        self.blk_emb = nn.Embedding(len(emb), 768, padding_idx=-1)
        self.blk_emb.weight.data.copy_(torch.from_numpy(emb))
        freeze_net(self.blk_emb)
        self.blk_trans = nn.Sequential(nn.Linear(768, 2 * d_model), nn.ReLU(), nn.Dropout(dropout_emb),
                                       nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.target_disease_vectors = torch.from_numpy(np.load(target_disease_path)).to(device)
        self.target_disease_att = Attention(d_model, target_att_heads, dropout)
        self.att_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False))
        self.mem_update = MemoryUpdate(d_model, dropout, self.blk_emb, self.blk_trans, mem_size)
        self.mem_size = mem_size
        self.mem_update_size = mem_update_size
        self.device = device
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(2 * d_model, 2)

    def forward(self, input_seqs, masks_ehr, input_txt, masks_txt, lengths, time_step, code_mask):
        v_all = self.ehr_encoder(input_seqs, lengths)
        target_disease_vecs = self.blk_trans(self.target_disease_vectors)
        target_disease_vecs = target_disease_vecs.repeat(v_all.size(0), 1, 1)
        transformed_v_all, _ = self.target_disease_att(v_all, target_disease_vecs, target_disease_vecs)

        v_text_all = self.blk_trans(self.blk_emb(input_txt))
        bs, seq_length, num_blk_pervisit, d_model = v_text_all.size()
        transformed_v_all_repeat = transformed_v_all.unsqueeze(2).expand(bs, seq_length, num_blk_pervisit, d_model)

        att_input = torch.cat((transformed_v_all_repeat, v_text_all), dim=3)
        att_score = self.att_mlp(att_input).view(bs, seq_length, num_blk_pervisit)
        att_score = torch.softmax(att_score, dim=-1)
        if masks_txt is not None:
            att_score = att_score.masked_fill(masks_txt == 0, 0)
        _, indices = torch.topk(att_score, self.mem_update_size, dim=2)
        selected_txt_ids = torch.gather(input_txt, dim=2, index=indices)
        selected_masks = torch.gather(masks_txt, dim=2, index=indices)

        memory = []
        mem_input = selected_txt_ids[:, 0].view(bs, self.mem_update_size)
        mem_mask_input = selected_masks[:, 0].view(bs, self.mem_update_size)
        for i in range(seq_length):
            if mem_input.size(1) > self.mem_size:
                query = transformed_v_all[:, i] + v_all[:, i]
                mem_output, mem_mask_output = self.mem_update(mem_input, mem_mask_input, query.view(bs, d_model))
            else:
                mem_output = mem_input
                mem_mask_output = mem_mask_input
            memory.append(mem_output)
            if i < seq_length - 1:
                mem_input = torch.cat((mem_output, selected_txt_ids[:, i + 1].view(bs, self.mem_update_size)), dim=1)
                mem_mask_input = torch.cat((mem_mask_output, selected_masks[:, i + 1].view(bs, self.mem_update_size)),
                                           dim=1)
        v_final = self.pooler(transformed_v_all, lengths)
        memory_final = torch.stack(memory[1:], dim=1).gather(1, lengths[:, None, None].expand(bs, 1,
                                                                                              self.mem_size) - 2).squeeze().view(
            bs, self.mem_size)
        mem_vec_final = self.pooler(self.blk_trans(self.blk_emb(memory_final)))
        output = self.output_mlp(torch.cat((v_final, mem_vec_final), dim=1))
        return output, memory_final


if __name__ == '__main__':
    lengths = torch.randint(0, 10, (16,))
    print(lengths)
    lengths = lengths[:, None, None].expand(16, 1, 5) - 2
    print(lengths)
