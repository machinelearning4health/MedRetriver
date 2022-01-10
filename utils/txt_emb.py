import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import pickle

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")


def make_block_vocab(in_path, outpath):
    input_jsonl = pickle.load(open(in_path, 'rb'))
    blocks = []
    for disease, desc in input_jsonl.items():
        for k, v in desc.items():
            blocks.extend(v)
    with open(outpath, 'w', encoding='utf-8') as fout:
        for bl in blocks:
            fout.write(bl + '\n')


def make_embedding(vocab_path, out_path):
    with open(vocab_path, 'r', encoding='utf-8') as fin:
        lines = [line for line in fin]
    model.eval()
    model.cuda()
    emb = []
    with torch.no_grad():
        for i in tqdm(range(len(lines))):
            encoding = tokenizer.encode_plus(lines[i])
            outputs = model(torch.tensor([encoding['input_ids']]).cuda(),
                            torch.tensor([encoding['token_type_ids']]).cuda())
            # print(outputs[0][0][0].size())
            emb.append(outputs[0][0][0].cpu().numpy())
    emb = np.array(emb)
    np.save(out_path, emb)


def add_padding(inpath, outpath):
    inarray = np.load(inpath)
    padding = np.zeros((1, inarray.shape[1]))
    outarray = np.concatenate((inarray, padding), axis=0)
    np.save(outpath, outarray)


def make_target_disease_vectors(disease_name, in_path, out_path):
    input_jsonl = pickle.load(open(in_path, 'rb'))
    blocks = []
    for k, v in input_jsonl[disease_name].items():
        blocks.extend(v)
    print(input_jsonl[disease_name])
    model.eval()
    model.cuda()
    emb = []
    with torch.no_grad():
        for i in tqdm(range(len(blocks))):
            encoding = tokenizer.encode_plus(blocks[i])
            outputs = model(torch.tensor([encoding['input_ids']]).cuda(),
                            torch.tensor([encoding['token_type_ids']]).cuda())
            # print(outputs[0][0][0].size())
            emb.append(outputs[0][0][0].cpu().numpy())
    emb = np.array(emb)
    np.save(out_path, emb)


if __name__ == "__main__":
    make_block_vocab('../data/processed/converted.pickle', '../data/processed/block_vocab.txt')
    make_embedding('../data/processed/block_vocab.txt', '../data/processed/block_embedding.npy')
    add_padding('../data/processed/block_embedding.npy', '../data/processed/block_embedding.npy')
    make_target_disease_vectors('Dementia', '../data/processed/converted.pickle', '../data/processed/dementia.npy')
    make_target_disease_vectors('COPD', '../data/processed/converted.pickle', '../data/processed/COPD.npy')
    make_target_disease_vectors('Heart failure', '../data/processed/converted.pickle', '../data/processed/heart_failure.npy')
    make_target_disease_vectors('Chronic kidney disease', '../data/processed/converted.pickle', '../data/processed/kidney_disease.npy')
    make_target_disease_vectors('Amnesia', '../data/processed/converted.pickle', '../data/processed/amnesia.npy')
