# 2021/11/27
# bfsujason@163.com

"""
Usage:

python bin/bert_align.py \
  -s data/mac/dev/zh \
  -t data/mac/dev/en \
  -o data/mac/dev/auto \
  -m data/mac/dev/meta_data.tsv \
  --src_embed data/mac/dev/zh/overlap data/mac/dev/zh/overlap.emb \
  --tgt_embed data/mac/dev/en/overlap data/mac/dev/en/overlap.emb \
  --max_align 8 --margin
"""

import os
import sys
import time
import torch
import faiss
import shutil
import argparse
import numpy as np
import numba as nb

def main():
  # user-defined parameters
  parser = argparse.ArgumentParser('Sentence alignment using Vecalign')
  parser.add_argument('-s', '--src', type=str, required=True, help='preprocessed source file to align')
  parser.add_argument('-t', '--tgt', type=str, required=True, help='preprocessed target file to align')
  parser.add_argument('-o', '--out', type=str, required=True, help='Output directory.')
  parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file.')
  parser.add_argument('--src_embed', type=str, nargs=2, required=True,
            help='Source embeddings. Requires two arguments: first is a text file, sencond is a binary embeddings file. ')
  parser.add_argument('--tgt_embed', type=str, nargs=2, required=True,
            help='Target embeddings. Requires two arguments: first is a text file, sencond is a binary embeddings file. ')
  parser.add_argument('--max_align', type=int, default=5, help='Maximum alignment types, n + m <= this value.')
  parser.add_argument('--win', type=int, default=5, help='Window size for the second-pass alignment.')
  parser.add_argument('--top_k', type=int, default=3, help='Top-k target neighbors of each source sentence.')
  parser.add_argument('--skip', type=float, default=-0.1, help='Similarity score for 0-1 and 1-0 alignment.')
  parser.add_argument('--margin', action='store_true', help='Margin-based cosine similarity')
  args = parser.parse_args()
  
  # fixed parameters to determine the
  # window size for the first-pass alignment  
  min_win_size = 10
  max_win_size = 600
  win_per_100 = 8

  # read in embeddings
  src_sent2line, src_line_embeddings = read_in_embeddings(args.src_embed[0], args.src_embed[1])
  tgt_sent2line, tgt_line_embeddings = read_in_embeddings(args.tgt_embed[0], args.tgt_embed[1])
  embedding_size = src_line_embeddings.shape[1]
  
  make_dir(args.out)
  jobs = create_jobs(args.meta, args.src, args.tgt, args.out)

  # start alignment
  for rec in jobs:
    src_file, tgt_file, align_file = rec.split("\t")
    print("Aligning {} to {}".format(src_file, tgt_file))
   
    # read in source and target sentences
    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
    
    # convert source and target texts into embeddings
    # and calculate sentence length
    t_0 = time.time()
    src_vecs, src_lens = doc2feats(src_sent2line, src_line_embeddings, src_lines, args.max_align - 1)
    tgt_vecs, tgt_lens = doc2feats(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.max_align - 1)
    char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])
    print("Reading embeddings takes {:.3f}".format(time.time() - t_0))
    
    # using faiss, find in the target text
    # the k nearest neighbors of each source sentence
    t_1 = time.time()
    if torch.cuda.is_available(): # GPU version
      res = faiss.StandardGpuResources() 
      index = faiss.IndexFlatIP(embedding_size)
      gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
      gpu_index.add(tgt_vecs[0,:]) 
      xq = src_vecs[0,:]
      D,I = gpu_index.search(xq,args.top_k)
    else: # CPU version
      index = faiss.IndexFlatIP(embedding_size) # use inter product to build index
      index.add(tgt_vecs[0,:])
      xq = src_vecs[0,:]
      D,I = index.search(xq, args.top_k)
    print("Finding top-k neighbors takes {:.3f}".format(time.time() - t_1))

    # find 1-to-1 alignment
    t_2 = time.time()
    src_len = len(src_lines)
    tgt_len = len(tgt_lines)
    first_alignment_types = make_alignment_types(2) # 0-0， 1-0 and 1-1
    first_w, first_search_path = find_first_search_path(src_len, tgt_len, min_win_size, max_win_size, win_per_100)
    first_pointers = first_pass_align(src_len, tgt_len, first_w, first_search_path, first_alignment_types, D, I, args.top_k)
    first_alignment = first_back_track(src_len, tgt_len, first_pointers, first_search_path, first_alignment_types)
    print("First pass alignment takes {:.3f}".format(time.time() - t_2))

    # find m-to-n alignment
    t_3 = time.time()
    second_w, second_search_path = find_second_search_path(first_alignment, args.win, src_len, tgt_len)
    second_alignment_types = make_alignment_types(args.max_align)
    second_pointers = second_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens, second_w, second_search_path, second_alignment_types, char_ratio, args.skip, margin=args.margin)
    second_alignment = second_back_track(src_len, tgt_len, second_pointers, second_search_path, second_alignment_types)
    print("Second pass alignment takes {:.3f}".format(time.time() - t_3))

    # save alignment
    print_alignments(second_alignment, align_file)
    
def second_back_track(i, j, b, search_path, a_types):
  alignment = []
  while ( i !=0 and j != 0 ):
    j_offset = j - search_path[i][0]
    a = b[i][j_offset]
    s = a_types[a][0]
    t = a_types[a][1]
    src_range = [i - offset - 1 for offset in range(s)][::-1]
    tgt_range = [j - offset - 1 for offset in range(t)][::-1]
    alignment.append((src_range, tgt_range))

    i = i-s
    j = j-t

  return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def second_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens, w, search_path, align_types, char_ratio, skip, margin=False):
  src_len = src_vecs.shape[1]
  tgt_len = tgt_vecs.shape[1]

  # intialize sum matrix
  cost = np.zeros((src_len + 1, w))
  #back = np.zeros((tgt_len + 1, w), dtype=nb.int64)
  back = np.zeros((src_len + 1, w), dtype=nb.int64)
  cost[0][0] = 0
  back[0][0] = -1
  
  for i in range(1, src_len + 1):
    i_start = search_path[i][0]
    i_end = search_path[i][1]
    for j in range(i_start, i_end + 1):
      if i + j == 0:
        continue
      best_score = -np.inf
      best_a = -1
      for a in range(align_types.shape[0]):
        a_1 = align_types[a][0]
        a_2 = align_types[a][1]
        prev_i = i - a_1
        prev_j = j - a_2

        if prev_i < 0 or prev_j < 0 :  # no previous cell in DP table 
          continue
        prev_i_start = search_path[prev_i][0]
        prev_i_end =  search_path[prev_i][1]
        if prev_j < prev_i_start or prev_j > prev_i_end: # out of bound of cost matrix
            continue
        prev_j_offset = prev_j - prev_i_start
        score = cost[prev_i][prev_j_offset]
        if score == -np.inf:
          continue

        if a_1 == 0 or a_2 == 0:  # deletion or insertion
          cur_score = skip
        else:
          src_v = src_vecs[a_1-1,i-1,:]
          tgt_v = tgt_vecs[a_2-1,j-1,:]
          src_l = src_lens[a_1-1, i-1]
          tgt_l = tgt_lens[a_2-1, j-1]
          cur_score = get_score(src_v, tgt_v, a_1, a_2, i, j, src_vecs, tgt_vecs, src_len, tgt_len, margin=margin)
          tgt_l = tgt_l * char_ratio
          min_len = min(src_l, tgt_l)
          max_len = max(src_l, tgt_l)
          len_p = np.log2(1 + min_len / max_len)
          cur_score *= len_p
        
        score += cur_score
        if score > best_score:
          best_score = score
          best_a = a

      j_offset = j - i_start
      cost[i][j_offset] = best_score
      back[i][j_offset] = best_a
      
  return back

@nb.jit(nopython=True, fastmath=True, cache=True)
def get_score(src_v, tgt_v, a_1, a_2, i, j, src_vecs, tgt_vecs, src_len, tgt_len, margin=False):
  similarity = nb_dot(src_v, tgt_v)
  if margin:
    tgt_neighbor_ave_sim = get_neighbor_sim(src_v, a_2, j, tgt_len, tgt_vecs)
    src_neighbor_ave_sim = get_neighbor_sim(tgt_v, a_1, i, src_len, src_vecs)
    neighbor_ave_sim = (tgt_neighbor_ave_sim + src_neighbor_ave_sim)/2
    similarity -= neighbor_ave_sim
    
  return similarity

@nb.jit(nopython=True, fastmath=True, cache=True)
def get_neighbor_sim(vec, a, j, len, db):
  left_idx = j - a
  right_idx = j + 1
    
  if right_idx > len:
    neighbor_right_sim = 0
  else:
    right_embed = db[0,right_idx-1,:]
    neighbor_right_sim = nb_dot(vec, right_embed)
    
  if left_idx == 0:
    neighbor_left_sim = 0
  else:
    left_embed = db[0,left_idx-1,:]
    neighbor_left_sim = nb_dot(vec, left_embed)

  #if right_idx > LEN or left_idx < 0:
  if right_idx > len or left_idx == 0:
    neighbor_ave_sim = neighbor_left_sim + neighbor_right_sim
  else:
    neighbor_ave_sim = (neighbor_left_sim + neighbor_right_sim) / 2
  
  return neighbor_ave_sim

@nb.jit(nopython=True, fastmath=True, cache=True)
def nb_dot(x, y):
  return np.dot(x,y)

def find_second_search_path(align, w, src_len, tgt_len):
  '''
  Convert 1-1 alignment from first-pass to the path for second-pass alignment.
  The index along X-axis and Y-axis must be consecutive. 
  '''
  last_bead_src = align[-1][0]
  last_bead_tgt = align[-1][1]
  
  if last_bead_src != src_len:
    if last_bead_tgt == tgt_len:
      align.pop()
    align.append((src_len, tgt_len))
  else:
    if last_bead_tgt != tgt_len:
      align.pop()
      align.append((src_len, tgt_len))
      
  prev_src, prev_tgt = 0,0
  path = []
  max_w = -np.inf
  for src, tgt in align:
    lower_bound = max(0, prev_tgt - w)
    upper_bound = min(tgt_len, tgt + w)
    path.extend([(lower_bound, upper_bound) for id in range(prev_src+1, src+1)])
    prev_src, prev_tgt = src, tgt
    width = upper_bound - lower_bound
    if width > max_w:
      max_w = width
  path = [path[0]] + path
  
  return max_w + 1, np.array(path)

def first_back_track(i, j, b, search_path, a_types):
  alignment = []
  while ( i !=0  and j != 0 ):
    j_offset = j - search_path[i][0]
    a = b[i][j_offset]
    s = a_types[a][0]
    t = a_types[a][1]
    if a == 2:
      alignment.append((i, j))

    i = i-s
    j = j-t
        
  return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def first_pass_align(src_len, tgt_len, w, search_path, align_types, dist, index, top_k):

  #initialize cost and backpointer matrix
  cost = np.zeros((src_len + 1, 2 * w + 1))
  pointers = np.zeros((src_len + 1, 2 * w + 1), dtype=nb.int64)
  cost[0][0] = 0
  pointers[0][0] = -1

  for i in range(1, src_len +  1):
    i_start = search_path[i][0]
    i_end = search_path[i][1]
    for j in range(i_start, i_end + 1):
      if i + j == 0:
        continue
      best_score = -np.inf
      best_a = -1
      for a in range(align_types.shape[0]):
        a_1 = align_types[a][0]
        a_2 = align_types[a][1]
        prev_i = i - a_1
        prev_j = j - a_2
        if prev_i < 0 or prev_j < 0 :  # no previous cell 
          continue
        prev_i_start = search_path[prev_i][0]
        prev_i_end =  search_path[prev_i][1]
        if prev_j < prev_i_start or prev_j > prev_i_end: # out of bound of cost matrix
            continue
        prev_j_offset = prev_j - prev_i_start
        score = cost[prev_i][prev_j_offset]
        if score == -np.inf:
          continue

        if a_1 > 0 and a_2 > 0:
          for k in range(top_k):
            if index[i-1][k] == j - 1:
              score += dist[i-1][k]
        if score > best_score:
          best_score = score
          best_a = a

      j_offset = j - i_start
      cost[i][j_offset] = best_score
      pointers[i][j_offset] = best_a

  return pointers

@nb.jit(nopython=True, fastmath=True, cache=True)
def find_first_search_path(src_len, tgt_len, min_win_size, max_win_size, win_per_100):
  yx_ratio = tgt_len / src_len
  win_size_1 = int(yx_ratio * tgt_len * win_per_100 / 100)
  win_size_2 = int(abs(tgt_len - src_len) * 3/4)
  w_1 = min(max(min_win_size, max(win_size_1, win_size_2)), max_win_size)
  w_2 = int(max(src_len, tgt_len) * 0.06)
  w = max(w_1, w_2)
  search_path = np.zeros((src_len + 1, 2), dtype=nb.int64)
  for i in range(0, src_len + 1):
    center = int(yx_ratio * i)
    w_start = max(0, center - w)
    w_end = min(center + w, tgt_len)
    search_path[i] = [w_start, w_end]
    
  return w, search_path

def doc2feats(sent2line, line_embeddings, lines, num_overlaps):
  lines = [preprocess_line(line) for line in lines]
  vecsize = line_embeddings.shape[1]
  vecs0 = np.empty((num_overlaps, len(lines), vecsize), dtype=np.float32)
  vecs1 = np.empty((num_overlaps, len(lines)), dtype=np.int)

  for ii, overlap in enumerate(range(1, num_overlaps + 1)):
    for jj, out_line in enumerate(layer(lines, overlap)):
      try:
        line_id = sent2line[out_line]
      except KeyError:
        logger.warning('Failed to find overlap=%d line "%s". Will use random vector.', overlap, out_line)
        line_id = None

      if line_id is not None:
        vec = line_embeddings[line_id]
      else:
        vec = np.random.random(vecsize) - 0.5
        vec = vec / np.linalg.norm(vec)
        
      vecs0[ii, jj, :] = vec
      vecs1[ii, jj] = len(out_line.encode("utf-8"))

  return vecs0, vecs1

def preprocess_line(line):
  line = line.strip()
  if len(line) == 0:
    line = 'BLANK_LINE'
    
  return line

def layer(lines, num_overlaps, comb=' '):
  """
  make front-padded overlapping sentences
  """
  if num_overlaps < 1:
    raise Exception('num_overlaps must be >= 1')
  out = ['PAD', ] * min(num_overlaps - 1, len(lines))
  for ii in range(len(lines) - num_overlaps + 1):
    out.append(comb.join(lines[ii:ii + num_overlaps]))
    
  return out

def read_in_embeddings(text_file, embed_file):
  sent2line = dict()
  with open(text_file, 'rt', encoding="utf-8") as fin:
    for ii, line in enumerate(fin):
      if line.strip() in sent2line:
        raise Exception('got multiple embeddings for the same line')
      sent2line[line.strip()] = ii

  line_embeddings = np.fromfile(embed_file, dtype=np.float32, count=-1)
  if line_embeddings.size == 0:
    raise Exception('Got empty embedding file')

  embedding_size = line_embeddings.size // len(sent2line)
  line_embeddings.resize(line_embeddings.shape[0] // embedding_size, embedding_size)
  
  return sent2line, line_embeddings

def make_alignment_types(max_alignment_size):
  # Return list of all (n,m) where n+m <= this
  alignment_types = []
  for x in range(1, max_alignment_size):
    for y in range(1, max_alignment_size):
      if x + y <= max_alignment_size:
        alignment_types.append([x, y])
  alignment_types = [[0,1], [1,0]] + alignment_types
  
  return np.array(alignment_types)

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
  print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
