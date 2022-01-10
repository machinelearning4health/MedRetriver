import re
import string
import os
import pickle
import bs4
from bs4 import BeautifulSoup as bs, NavigableString
from tqdm import tqdm

filenames = os.listdir('F:/mayo-clinic-scraper/disease_condition/')


def proc_file(fname, out):
    with open('F:/mayo-clinic-scraper/disease_condition/' + fname, 'r', encoding='utf-8') as fin:
        content = bs(fin.read(), 'html.parser')
        to_del = content.find_all("h3")
        for ele in to_del:
            ele.extract()

        titles = content.find_all(re.compile('^h2'))
        paras = content.find_all("p")
        uls = content.find_all("ul")

        child_nodes = content.div.contents
        for node in child_nodes[:]:
            if node not in titles + paras + uls:
                child_nodes.remove(node)

        title_idxs = []
        for i in range(len(child_nodes)):
            if child_nodes[i] in titles:
                title_idxs.append(i)

        results = {}
        for j in range(len(title_idxs) - 1):
            results[child_nodes[title_idxs[j]].text] = [child_nodes[k] for k in range(title_idxs[j] + 1, title_idxs[j + 1]) if
                                                  len(child_nodes[k].text.strip()) > 0]
        keys_todel = []
        for item in results.items():
            if item[0] not in ['Risk factors', 'Causes', 'Complications', 'Symptoms']:
                keys_todel.append(item[0])
        for key in keys_todel:
            results.pop(key)

        for key, item in results.items():
            value = []

            p_ids = []
            ul_ids = []
            for i in range(len(item)):
                if item[i] in paras:
                    p_ids.append(i)
                elif item[i] in uls:
                    ul_ids.append(i)
                else:
                    print(item[i])

            pre_v = []
            for k in range(len(item)):
                if k in p_ids:
                    pre_v.append(item[k].text.strip())
                elif k in ul_ids:
                    tmp = [li.text.strip() for li in item[k].contents if not isinstance(li, NavigableString) and len(li.text.strip()) > 0]
                    if tmp[0][-1] == '.':
                        pre_v.append(tmp)
                    else:
                        pre_v.append(', '.join(tmp) + '.')

            spec_pos = []
            for l in range(len(pre_v) - 1):
                if l in p_ids and l+1 in ul_ids and isinstance(pre_v[l+1], str):
                    spec_pos.append(l+1)
            for l in range(len(pre_v)):
                if l in spec_pos:
                    continue
                else:
                    if l < len(pre_v) - 1 and l in p_ids and l + 1 in ul_ids and isinstance(pre_v[l + 1], str):
                        value.append(pre_v[l] + ' ' + pre_v[l+1])
                    else:
                        if isinstance(pre_v[l], list):
                            value.extend(pre_v[l])
                        else:
                            value.append(pre_v[l])

            results[key] = value
        if results:
            out[fname.strip('.txt')] = results

    return out

if __name__ == "__main__":
    with open('../data/processed/converted.pickle', 'wb') as fout:
        out = {}
        for fname in tqdm(filenames, total=len(filenames)):
            out = proc_file(fname, out)
        print(len(out))
        pickle.dump(out, fout)
