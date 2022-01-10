from fuzzywuzzy import fuzz
import pandas as pd
import pickle
import os
from tqdm import tqdm

icd2des = {}
df = pd.read_excel('../data/CMS32_DESC_LONG_SHORT_DX.xlsx')
icds = df['DIAGNOSIS CODE']
desc = df['LONG DESCRIPTION']
for i in range(len(icds)):
    icd = icds[i][0:3] + '.' + icds[i][3:]
    icd2des[icd] = desc[i]

with open('../data/amnesia/amnesia_code2idx_new.pickle', 'rb') as fin:
    code2idx = pickle.load(fin)
print(len(code2idx))
files = os.listdir('F:\\mayo-clinic-scraper\\disease_condition')
topics = [file.strip('.txt') for file in files]

code2topic = {}
for code in tqdm(code2idx.keys(), total=len(code2idx)):
    if code in icd2des.keys():
        max_ratio = 0
        topic_map = ''
        for topic in topics:
            if fuzz.token_set_ratio(icd2des[code], topic) > max_ratio:
                max_ratio = fuzz.token_set_ratio(icd2des[code], topic)
                topic_map = topic
        if max_ratio > 90:
            code2topic[code] = topic_map

print(code2topic)
print(len(code2topic))

with open('../data/amnesia/amnesia_code2topic.pickle', 'wb') as fout:
    pickle.dump(code2topic, fout)