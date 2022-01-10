import re
import string
import os
import pickle
import bs4
from bs4 import BeautifulSoup as bs, NavigableString
from tqdm import tqdm
filenames = os.listdir('./data/disease_condition')
out = {}
with open('./data/processed/converted.pickle', 'wb') as fout:
    for fname in tqdm(filenames, total=len(filenames)):
        with open('./data/disease_condition/' + fname, 'r', encoding='utf-8') as fin:
            content = bs(fin.read(), 'html.parser')
            titles = content.find_all(re.compile('^h'))
            child_nodes = content.div.contents
            for node in child_nodes:
                if isinstance(node, NavigableString):
                    child_nodes.remove(node)
            idxs = []
            for i in range(len(child_nodes)):
                if child_nodes[i] in titles:
                    idxs.append(i)
            results = {}
            for j in range(len(idxs) - 1):
                results[child_nodes[idxs[j]].text] = [child_nodes[k] for k in range(idxs[j] + 1, idxs[j + 1])]
            keys_todel = []
            for item in results.items():
                if item[0] not in ['Risk factors', 'Causes', 'Complications', 'Symptoms']:
                    keys_todel.append(item[0])
            for key in keys_todel:
                results.pop(key)
            #print(results)
            for item in results.items():
                value = []
                for k in range(len(item[1])):
                    if item[1][k].find("li"):
                        tmp = [li.text.strip() for li in item[1][k].contents if not isinstance(li, NavigableString)]
                        if tmp[0][-1] == '.':
                            value.extend(tmp)
                        else:
                            if value:
                                value.pop(-1)
                                sent = item[1][k-1].text.strip() + ' ' + ', '.join(tmp) + '.'
                                if len(sent) > 10:
                                    value.append(sent)
                            else:
                                sent = ', '.join(tmp) + '.'
                                if len(sent) > 10:
                                    value.append(sent)
                    else:
                        sent = item[1][k].text.strip()
                        if len(sent) > 10:
                            value.append(sent)
                results[item[0]] = value
            if results:
                out[fname.strip('.txt')] = results
            #print(results)
    pickle.dump(out, fout)
