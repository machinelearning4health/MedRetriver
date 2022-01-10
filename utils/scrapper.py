import requests
import re
import string
import os
from bs4 import BeautifulSoup as bs
from tqdm import tqdm


disease_dic = {}
with open('disease_urls.txt', 'r') as fin:
    urls = [line.strip() for line in fin]
    for i in tqdm(range(0, len(urls))):
        req = requests.get(urls[i],
                           headers={'authority': 'www.mayoclinic.org', 'user-agent': 'Mozilla/5.0 (Linux; Android '
                                                                                     '6.0; Nexus 5 Build/MRA58N) '
                                                                                     'AppleWebKit/537.36 (KHTML, '
                                                                                     'like Gecko) '
                                                                                     'Chrome/85.0.4183.121 Mobile '
                                                                                     'Safari/537.36'}).text
        sp = bs(req, "html.parser")
        title = sp.find("div", class_="main").header.find("h1").text
        content = sp.find("article", id="main-content").find("div", class_="content").find("div", class_=None)
        to_del = content.find_all("div")
        for div in to_del:
            div.extract()
        with open('./disease_condition/' + title.replace('/', '／').replace('\\', '＼') + '.txt', 'w',
                  encoding='utf-8') as fout:
            fout.write(str(content))
