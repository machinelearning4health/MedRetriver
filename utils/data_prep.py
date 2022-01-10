import pickle
import os
import json
from tqdm import tqdm
import numpy as np


def match_code2sequence(code, code2topic, topic2des, blk2idx):
    topic = code2topic[code]
    seq = []
    if topic in topic2des.keys():
        description = topic2des[topic]
        for key, value in description.items():
            for blk in value:
                if blk in blk2idx.keys():
                    seq.append(blk2idx[blk])
    return seq


def prep_dataset(data_path, code2id_path, code2topic_path, topic2des_path, vocab_path, out_path):
    with open(data_path, 'rb') as fin, open(code2id_path, 'rb') as fin2, open(code2topic_path, 'rb') as fin3, open(
            topic2des_path, 'rb') as fin4:
        ehr = pickle.load(fin)
        code2id = pickle.load(fin2)
        code2topic = pickle.load(fin3)
        topic2des = pickle.load(fin4)

    with open(vocab_path, 'r', encoding='utf-8') as fin5:
        vocab = [line.strip() for line in fin5]
    blk2idx = {blk: i for i, blk in enumerate(vocab)}

    id2code = {v: k for k, v in code2id.items()}
    nrow = len(ehr[1])
    txt_ids = []
    for i in tqdm(range(nrow)):
        seq = []
        for visit in ehr[0][i]:
            seq4vis = []
            for id in visit:
                code = id2code[id]
                if code in code2topic.keys():
                    seq4code = match_code2sequence(code, code2topic, topic2des, blk2idx)
                    seq4vis.extend(seq4code)
            seq.append(seq4vis)
        txt_ids.append(seq)
    with open(out_path, 'wb') as fout:
        pickle.dump(txt_ids, fout)


def stat(path):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    maxlens = []
    for patient in data:
        txt_lens = [len(vis) for vis in patient if len(vis) > 0]
        maxlens.extend(txt_lens)
    lens = np.array(maxlens)
    pert = np.percentile(lens, 95)
    mlen = max(maxlens)
    avg = sum(maxlens) / len(maxlens)
    print(mlen, avg, pert)


if __name__ == "__main__":

    stat('../data/amnesia/amnesia_training_txt.pickle')
    stat('../data/kidney/kidney_training_txt.pickle')
    stat('../data/hf/hf_training_txt.pickle')
    stat('../data/dementia/dementia_training_txt.pickle')
    stat('../data/copd/copd_training_txt.pickle')