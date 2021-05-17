import os
import argparse
import time
import numpy as np
from sentence_transformers import SentenceTransformer

def _main():
    parser = argparse.ArgumentParser('Convert sentences into vectors.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, help='input file.')
    parser.add_argument('-o', '--output', type=str, help='output file.')
    args = parser.parse_args()
    model = SentenceTransformer('LaBSE')
    embed_texts(model, args.input, args.output)

def get_sents(TEXT):
    sents = []
    with open(TEXT, 'r', encoding="utf-8") as f:
        for line in f:
            sents.append(line.strip())
    return sents

def embed_texts(model, fin, fout):
    t_0 = time.time()
    print("Embedding text {} ...".format(fin))
    txt = get_sents(fin)
    embed = model.encode(txt)
    embed.tofile(fout)
    print("It takes {} seconds".format(time.time() - t_0))
    
if __name__ == '__main__':
    _main()
