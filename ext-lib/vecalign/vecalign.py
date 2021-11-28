#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Usage:

python ext-lib/vecalign/vecalign.py \
  -s data/mac/dev/zh \
  -t data/mac/dev/en \
  -o data/mac/dev/auto \
  -m data/mac/dev/meta_data.tsv \
  --src_embed data/mac/dev/zh/overlap data/mac/dev/zh/overlap.emb \
  --tgt_embed data/mac/dev/en/overlap data/mac/dev/en/overlap.emb \
  -a 8 -v
"""

import os
import time
import argparse
import shutil
import logging
import pickle
from math import ceil
from random import seed as seed

import numpy as np

logger = logging.getLogger('vecalign')
logger.setLevel(logging.WARNING)
logFormatter = logging.Formatter("%(asctime)s  %(levelname)-5.5s  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

from dp_utils import make_alignment_types, read_alignments, read_in_embeddings, make_doc_embedding, vecalign

def main():
  # make runs consistent
  seed(42)
  np.random.seed(42)

  parser = argparse.ArgumentParser('Sentence alignment using Vecalign')
  parser.add_argument('-s', '--src', type=str, required=True,
            help='preprocessed source file to align')
  parser.add_argument('-t', '--tgt', type=str, required=True,
            help='preprocessed target file to align')
  parser.add_argument('-o', '--out', type=str, required=True,
            help='Output directory.')
  parser.add_argument('-m', '--meta', type=str, required=True,
            help='Metadata file.')
  parser.add_argument('--src_embed', type=str, nargs=2, required=True,
            help='Source embeddings. Requires two arguments: first is a text file, sencond is a binary embeddings file. ')
  parser.add_argument('--tgt_embed', type=str, nargs=2, required=True,
            help='Target embeddings. Requires two arguments: first is a text file, sencond is a binary embeddings file. ')
  parser.add_argument('-a', '--alignment_max_size', type=int, default=5,
            help='Searches for alignments up to size N-M, where N+M <= this value. Note that the the embeddings must support the requested number of overlaps')
  parser.add_argument('-d', '--del_percentile_frac', type=float, default=0.2,
            help='Deletion penalty is set to this percentile (as a fraction) of the cost matrix distribution. Should be between 0 and 1.')
  parser.add_argument('-v', '--verbose', help='sets consle to logging.DEBUG instead of logging.WARN',
            action='store_true')
  parser.add_argument('--max_size_full_dp', type=int, default=300,
            help='Maximum size N for which is is acceptable to run full N^2 dynamic programming.')
  parser.add_argument('--costs_sample_size', type=int, default=20000,
            help='Sample size to estimate costs distribution, used to set deletion penalty in conjunction with deletion_percentile.')
  parser.add_argument('--num_samps_for_norm', type=int, default=100,
            help='Number of samples used for normalizing embeddings')
  parser.add_argument('--search_buffer_size', type=int, default=5,
            help='Width (one side) of search buffer. Larger values makes search more likely to recover from errors but increases runtime.')
  args = parser.parse_args()

  if args.verbose:
    import logging
    logger.setLevel(logging.INFO)

  if args.alignment_max_size < 2:
    logger.warning('Alignment_max_size < 2. Increasing to 2 so that 1-1 alignments will be considered')
    args.alignment_max_size = 2

  src_sent2line, src_line_embeddings = read_in_embeddings(args.src_embed[0], args.src_embed[1])
  tgt_sent2line, tgt_line_embeddings = read_in_embeddings(args.tgt_embed[0], args.tgt_embed[1])

  width_over2 = ceil(args.alignment_max_size / 2.0) + args.search_buffer_size

  make_dir(args.out)
  jobs = create_jobs(args.meta, args.src, args.tgt, args.out)
  
  for rec in jobs:
    src_file, tgt_file, align_file = rec.split("\t")
    logger.info('Aligning src="%s" to tgt="%s"', src_file, tgt_file)
    
    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
    vecs0 = make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, args.alignment_max_size)

    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
    vecs1 = make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.alignment_max_size)

    final_alignment_types = make_alignment_types(args.alignment_max_size)
    logger.debug('Considering alignment types %s', final_alignment_types)

    stack = vecalign(vecs0=vecs0,
             vecs1=vecs1,
             final_alignment_types=final_alignment_types,
             del_percentile_frac=args.del_percentile_frac,
             width_over2=width_over2,
             max_size_full_dp=args.max_size_full_dp,
             costs_sample_size=args.costs_sample_size,
             num_samps_for_norm=args.num_samps_for_norm)

    # write final alignments
    print_alignments(stack[0]['final_alignments'], align_file)

def create_jobs(meta, src, tgt, out):
  jobs = []
  fns = get_fns(meta)
  for file in fns:
    src_path = os.path.abspath(os.path.join(src, file))
    tgt_path = os.path.abspath(os.path.join(tgt, file))
  
    out_path = os.path.abspath(os.path.join(out, file + '.align'))
    jobs.append('\t'.join([src_path, tgt_path, out_path]))
    
  return jobs

def get_fns(meta):
  fns = []
  with open(meta, 'rt', encoding='utf-8') as f:
    next(f) # skip header
    for line in f:
      recs = line.strip().split('\t')
      fns.append(recs[0])

  return fns

def print_alignments(alignments, out):
  with open(out, 'wt', encoding='utf-8') as f:
    for x, y in alignments:
      f.write("{}:{}\n".format(x, y))

def make_dir(path):
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path, exist_ok=True)
  
if __name__ == '__main__':
  t_0 = time.time()
  main()
  print("It takes {} seconds to aligent all the sentences.".format(time.time() - t_0))
