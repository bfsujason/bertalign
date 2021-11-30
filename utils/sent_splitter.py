# 2021/11/30
# bfsujason@163.com

"""
Usage:

python utils/sent_splitter.py \
    -i utils/zh_raw
    -o utils/zh
    -l zh
"""

import os
import re
import shutil
import argparse
import pysbd

def main():
    parser = argparse.ArgumentParser(description='Split multilingual sentences using pySBD')
    parser.add_argument('-i', '--input', type=str, required=True, help='Directory for raw files.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Directory for split files.')
    parser.add_argument('-l', '--language', type=str, required=True, help='ISO 639-1 language code.')
    args = parser.parse_args()
    
    make_dir(args.output)
    splitter = pysbd.Segmenter(language=args.language, clean=False)
    for file in os.listdir(args.input):
        print("Splitting file {} ...".format(file))
        sents = split_sents(os.path.join(args.input, file), splitter)
        write_sents(os.path.join(args.output, file), sents)
  
def write_sents(fp, sents):
    with open(fp, 'wt', encoding='utf-8') as f:
        for sent in sents:
            f.write(sent + '\n')     

def split_sents(fp, splitter):
    paras = get_paras(fp)
    sents_in_para = []
    for para in paras:
        cur_sents = splitter.segment(para)
        sents_in_para.append(cur_sents)
    sents = [j for sub in sents_in_para for j in sub]
    return sents

def get_paras(fp):
    paras = []
    with open(fp, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                line = re.sub('\s+', ' ', line)
                paras.append(line)
    return paras
    
def make_dir(path):
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    main()
