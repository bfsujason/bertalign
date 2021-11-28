# 2021/11/27
# bfsujason@163.com

'''
Usage (Linux):

python bin/embed_sents.py \
  -i data/mac/dev/zh \
  -o data/mac/dev/zh/overlap data/mac/dev/zh/overlap.emb \
  -m data/mac/test/meta_data.tsv \
  -n 8
'''

import os
import time
import shutil
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
  parser = argparse.ArgumentParser(description='Multilingual sentence embeddings')
  parser.add_argument('-i', '--input', type=str, required=True, help='Data directory.')
  parser.add_argument('-o', '--output', type=str, required=True, nargs=2, help='Overalp and embedding file.')
  parser.add_argument('-n', '--num_overlaps', type=int, default=5, help='Maximum number of allowed overlaps.')
  parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file.')
  args = parser.parse_args()
  
  fns = get_fns(args.meta)
  overlap = get_overlap(args.input, fns, args.num_overlaps)
  write_overlap(overlap, args.output[0])
  
  model = load_model()
  embed_overlap(model, overlap, args.output[1])
  
def embed_overlap(model, overlap, fout):
  print("Embedding text ...")
  t_0 = time.time()
  embed = model.encode(overlap)
  embed.tofile(fout)
  print("It takes {:.3f} seconods to embed text.".format(time.time() - t_0))

def write_overlap(overlap, outfile):
  with open(outfile, 'wt', encoding="utf-8") as fout:
    for line in overlap:
      fout.write(line + '\n')

def get_overlap(dir, fns, n):
  overlap = set()
  for file in fns:
    in_path = os.path.join(dir, file)   
    lines = open(in_path, 'rt', encoding="utf-8").readlines()
    for out_line in yield_overlaps(lines, n):
      overlap.add(out_line)
    
  # for reproducibility
  overlap = list(overlap)
  overlap.sort()
  
  return overlap

def yield_overlaps(lines, num_overlaps):
  lines = [preprocess_line(line) for line in lines]
  for overlap in range(1, num_overlaps + 1):
    for out_line in layer(lines, overlap):
      # check must be here so all outputs are unique
      out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
      yield out_line2
      
def layer(lines, num_overlaps, comb=' '):
  if num_overlaps < 1:
    raise Exception('num_overlaps must be >= 1')
  out = ['PAD', ] * min(num_overlaps - 1, len(lines))
  for ii in range(len(lines) - num_overlaps + 1):
    out.append(comb.join(lines[ii:ii + num_overlaps]))
  return out
  
def preprocess_line(line):
  line = line.strip()
  if len(line) == 0:
    line = 'BLANK_LINE'
  return line

def load_model():
  print("Loading embedding model ...")
  t0 = time.time()
  model = SentenceTransformer('LaBSE')
  print("It takes {:.3f} seconods to load the model.".format(time.time() - t0))
  return model
  
def get_fns(meta):
  fns = []
  with open(meta, 'rt', encoding='utf-8') as f:
    next(f) # skip header
    for line in f:
      recs = line.strip().split('\t')
      fns.append(recs[0])

  return fns

def make_dir(path):
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path, exist_ok=True)
  
if __name__ == '__main__':
  main()
