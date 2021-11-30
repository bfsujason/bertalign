# 2021/11/29
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
import time
import torch
import faiss
import shutil
import argparse
import numpy as np
import numba as nb

def main():
  # user-defined parameters
  parser = argparse.ArgumentParser('Sentence alignment using Bertalign')
  parser.add_argument('-s', '--src', type=str, required=True, help='Source texts directory.')
  parser.add_argument('-t', '--tgt', type=str, required=True, help='Target texts directory.')
  parser.add_argument('-o', '--out', type=str, required=True, help='Alignment directory.')
  parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file path.')
  parser.add_argument('--src_embed', type=str, nargs=2, required=True,
            help='Source overlapping and embedding file paths.')
  parser.add_argument('--tgt_embed', type=str, nargs=2, required=True,
            help='Target overlapping and embedding file paths.')
  parser.add_argument('--max_align', type=int, default=5,
            help='Maximum number of source+target sentences allowed in each alignment segment.')
  parser.add_argument('--win', type=int, default=5, help='Window size for the second-pass alignment.')
  parser.add_argument('--top_k', type=int, default=3, help='Top-k target neighbors of each source sentence.')
  parser.add_argument('--skip', type=float, default=-0.1, help='Similarity score for 0-1 and 1-0 alignment.')
  parser.add_argument('--margin', action='store_true', help='Margin-based modified cosine similarity.')
  args = parser.parse_args()
  
  # Read in source and target embeddings.
  src_sent2line, src_line_embeddings = \
    read_in_embeddings(args.src_embed[0], args.src_embed[1])
  tgt_sent2line, tgt_line_embeddings = \
    read_in_embeddings(args.tgt_embed[0], args.tgt_embed[1])
  
  # Perform stentence alignment.
  make_dir(args.out)
  jobs = create_jobs(args.meta, args.src, args.tgt, args.out)
  for job in jobs:
    src_file, tgt_file, out_file = job.split('\t')
    print("Aligning {} to {}".format(src_file, tgt_file))

    # Convert source and target texts into feature matrix.
    t_0 = time.time()
    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
    src_vecs, src_lens = \
      doc2feats(src_sent2line, src_line_embeddings, src_lines, args.max_align - 1)
    tgt_vecs, tgt_lens = \
      doc2feats(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.max_align - 1)
    char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])
    print("Vectorizing soure and target texts takes {:.3f} seconds.".format(time.time() - t_0))

    # Find the top_k similar target sentences for each source sentence.
    t_1 = time.time()
    D, I = find_top_k_sents(src_vecs[0,:], tgt_vecs[0,:], k=args.top_k)
    print("Finding top-k sentences takes {:.3f} seconds.".format(time.time() - t_1))

    # Find optimal 1-1 alignments using dynamic programming.
    t_2 = time.time()
    m = len(src_lines)
    n = len(tgt_lines)
    first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
    first_w, first_path = find_first_search_path(m, n)
    first_pointers = first_pass_align(m, n, first_w,
                                      first_path, first_alignment_types,
                                      D, I, args.top_k)
    first_alignment = first_back_track(m, n,
                                       first_pointers, first_path,
                                       first_alignment_types)
    print("First-pass alignment takes {:.3f} seconds.".format(time.time() - t_2))
    
    # Find optimal m-to-n alignments using dynamic programming.
    t_3 = time.time()
    second_alignment_types = get_alignment_types(args.max_align)
    second_w, second_path = find_second_path(first_alignment, args.win, m, n)
    second_pointers = second_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens,
                                        second_w, second_path, second_alignment_types,
                                        char_ratio, args.skip, margin=args.margin)
    second_alignment = second_back_track(m, n, second_pointers,
                                         second_path, second_alignment_types)
    print("Second pass alignment takes {:.3f}".format(time.time() - t_3))

    # save alignment results
    print_alignments(second_alignment, out_file)

def print_alignments(alignments, out):
  with open(out, 'wt', encoding='utf-8') as f:
    for x, y in alignments:
      f.write("{}:{}\n".format(x, y))

@nb.jit(nopython=True, fastmath=True, cache=True)
def second_pass_align(src_vecs,
                      tgt_vecs,
                      src_lens,
                      tgt_lens,
                      w,
                      search_path,
                      align_types,
                      char_ratio,
                      skip,
                      margin=False):
  """
  Perform the second-pass alignment to extract n-m bitext segments.
  Args:
      src_vecs: numpy array of shape (max_align-1, num_src_sents, embedding_size).
      tgt_vecs: numpy array of shape (max_align-1, num_tgt_sents, embedding_size)
      src_lens: numpy array of shape (max_align-1, num_src_sents).
      tgt_lens: numpy array of shape (max_align-1, num_tgt_sents).
      w: int. Predefined window size for the second-pass alignment.
      search_path: numpy array. Second-pass alignment search path.
      align_types: numpy array. Second-pass alignment types.
      char_ratio: float. Ratio between source length to target length.
      skip: float. Cost for instertion and deletion.
      margin: boolean. Set to true if choosing modified cosine similarity score.
  Returns:
      pointers: numpy array recording best alignments for each DP cell.
  """
  src_len = src_vecs.shape[1]
  tgt_len = tgt_vecs.shape[1]

  # Intialize cost and backpointer matrix
  cost = np.zeros((src_len + 1, w))
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
          cur_score = get_score(src_v, tgt_v,
                                a_1, a_2, i, j,
                                src_vecs, tgt_vecs,
                                src_len, tgt_len,
                                margin=margin)
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

def second_back_track(i, j, b, search_path, a_types):
  alignment = []
  #while ( i !=0 and j != 0 ):
  while ( 1 ):
    j_offset = j - search_path[i][0]
    a = b[i][j_offset]
    s = a_types[a][0]
    t = a_types[a][1]
    src_range = [i - offset - 1 for offset in range(s)][::-1]
    tgt_range = [j - offset - 1 for offset in range(t)][::-1]
    alignment.append((src_range, tgt_range))

    i = i-s
    j = j-t
    
    if i == 0 and j == 0:
        return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def get_score(src_v, tgt_v,
              a_1, a_2,
              i, j,
              src_vecs, tgt_vecs,
              src_len, tgt_len,
              margin=False):
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

def find_second_path(align, w, src_len, tgt_len):
  '''
  Convert 1-1 alignment from first-pass to the path for second-pass alignment.
  The indices along X-axis and Y-axis must be consecutive.
  Args:
      align: list of tuples. First-pass alignment results.
      w: int. Predefined window size for the second path.
      src_len: int. Number of source sentences.
      tgt_len: int. Number of target sentences.
  Returns:
      path: numpy array for the second search path.
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
      
  prev_src, prev_tgt = 0, 0
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
  """
  Retrieve 1-1 alignments from the first-pass DP table.
  Args:
      i: int. Number of source sentences.
      j: int. Number of target sentences.
      search_path: numpy array. First-pass search path.
      a_types: numpy array. First-pass alignment types.
  Returns:
      alignment: list of tuples for 1-1 alignments.
  """
  alignment = []
  #while ( i !=0  and j != 0 ):
  while ( 1 ):
    j_offset = j - search_path[i][0]
    a = b[i][j_offset]
    s = a_types[a][0]
    t = a_types[a][1]
    if a == 2: # best 1-1 alignment
      alignment.append((i, j))

    i = i-s
    j = j-t
    
    if i == 0 and j == 0:
        return alignment[::-1]

@nb.jit(nopython=True, fastmath=True, cache=True)
def first_pass_align(src_len,
                     tgt_len,
                     w,
                     search_path,
                     align_types,
                     dist,
                     index,
                     top_k):
  """
  Perform the first-pass alignment to extract 1-1 bitext segments.
  Args:
      src_len: int. Number of source sentences.
      tgt_len: int. Number of target sentences.
      w: int. Window size for the first-pass alignment.
      search_path: numpy array. Search path for the first-pass alignment.
      align_types: numpy array. Alignment types for the first-pass alignment.
      dist: numpy array. Distance matrix for top-k similar vecs.
      index: numpy array. Index matrix for top-k similar vecs.
      top_k: int. Number of most similar top-k vecs.
  Returns:
      pointers: numpy array recording best alignments for each DP cell.
  """
  # Initialize cost and backpointer matrix.
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

def find_first_search_path(src_len,
                           tgt_len,
                           min_win_size = 250,
                           percent=0.06):
  """
  Find the window size and search path for the first-pass alignment.
  Args:
      src_len: int. Number of source sentences.
      tgt_len: int. Number of target sentences.
      min_win_size: int. Minimum window size.
      percent. float. Percent of longer sentences.
  Returns:
      win_size: int. Window size along the diagonal of the DP table.
      search_path: numpy array of shape (src_len + 1, 2), containing the start
                   and end index of target sentences for each source sentence.
                   One extra row is added in the search_path for calculation of
                   deletions and omissions.
  """
  win_size = max(min_win_size, int(max(src_len, tgt_len) * percent))
  search_path = []
  yx_ratio = tgt_len / src_len
  for i in range(0, src_len + 1):
    center = int(yx_ratio * i)
    win_start = max(0, center - win_size)
    win_end = min(center + win_size, tgt_len)
    search_path.append([win_start, win_end])
  return win_size, np.array(search_path)

def get_alignment_types(max_alignment_size):
  """
  Get all the possible alignment types.
  Args:
    max_alignment_size: int. Source sentences number +
                             Target sentences number <= this value.
  Returns:
    alignment_types: numpy array.
  """
  alignment_types = [[0,1], [1,0]]
  for x in range(1, max_alignment_size):
    for y in range(1, max_alignment_size):
      if x + y <= max_alignment_size:
        alignment_types.append([x, y])    
  return np.array(alignment_types)

def find_top_k_sents(src_vecs, tgt_vecs, k=3):
  """
  Find the top_k similar vecs in tgt_vecs for each vec in src_vecs.
  Args:
      src_vecs: numpy array of shape (num_src_sents, embedding_size)
      tgt_vecs: numpy array of shape (num_tgt_sents, embedding_size)
      k: int. Number of most similar target sentences.
  Returns:
      D: numpy array. Similarity score matrix of shape (num_src_sents, k).
      I: numpy array. Target index matrix of shape (num_src_sents, k).
  """
  embedding_size = src_vecs.shape[1]
  if torch.cuda.is_available(): # GPU version
    res = faiss.StandardGpuResources() 
    index = faiss.IndexFlatIP(embedding_size)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(tgt_vecs) 
    D, I = gpu_index.search(src_vecs, k)
  else: # CPU version
    index = faiss.IndexFlatIP(embedding_size)
    index.add(tgt_vecs)
    D, I = index.search(src_vecs, k)
  return D, I

def doc2feats(sent2line, line_embeddings, lines, num_overlaps):
  """
  Convert texts into feature matrix.
  Args:
      sent2line: dict. Map each sentence to its ID.
      line_embeddings: numpy array of sentence embeddings.
      lines: list of sentences.
      num_overlaps: int. Maximum number of overlapping sentences allowed.
  Returns:
      vecs0: numpy array of shape (num_overlaps, num_lines, size_embedding)
             for overlapping sentence embeddings.
      vecs1: numpy array of shape (num_overlap, num_lines)
             for overlapping sentence lengths.
  """
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

def layer(lines, num_overlaps, comb=' '):
  """
  Make front-padded overlapping sentences.
  """
  if num_overlaps < 1:
    raise Exception('num_overlaps must be >= 1')
  out = ['PAD', ] * min(num_overlaps - 1, len(lines))
  for ii in range(len(lines) - num_overlaps + 1):
    out.append(comb.join(lines[ii:ii + num_overlaps]))    
  return out

def preprocess_line(line):
  """
  Clean each line of the text.
  """
  line = line.strip()
  if len(line) == 0:
    line = 'BLANK_LINE'  
  return line

def read_in_embeddings(text_file, embed_file):
  """
  Read in the overlap lines and line embeddings.
  Args:
      text_file: str. Overlap file path.
      embed_file: str. Embedding file path.
  Returns:
      sent2line: dict. Map overlap sentences to line IDs.
      line_embeddings: numpy array of the shape (num_lines, embedding_size).
                       For sentence-transformers, the embedding_size is 768. 
  """
  sent2line = dict()
  with open(text_file, 'rt', encoding="utf-8") as f:
    for i, line in enumerate(f):
      sent2line[line.strip()] = i
  line_embeddings = np.fromfile(embed_file, dtype=np.float32)
  embedding_size = line_embeddings.size // len(sent2line)
  line_embeddings.resize(line_embeddings.shape[0] // embedding_size, embedding_size)
  return sent2line, line_embeddings

def create_jobs(meta_data_file, src_dir, tgt_dir, alignment_dir):
  """
  Creat a job list consisting of source, target and alignment file paths.
  """
  jobs = []
  text_ids = get_text_ids(meta_data_file)
  for id in text_ids:
    src_path = os.path.abspath(os.path.join(src_dir, id))
    tgt_path = os.path.abspath(os.path.join(tgt_dir, id))
    out_path = os.path.abspath(os.path.join(alignment_dir, id + '.align'))
    jobs.append('\t'.join([src_path, tgt_path, out_path]))  
  return jobs

def get_text_ids(meta_data_file):
  """
  Get the text IDs to be aligned.
  Args:
      meta_data_file: str. TSV file with the first column being text ID.
  Returns:
      text_ids: list.
  """
  text_ids = []
  with open(meta_data_file, 'rt', encoding='utf-8') as f:
    next(f) # skip header
    for line in f:
      recs = line.strip().split('\t')
      text_ids.append(recs[0])
  return text_ids

def make_dir(auto_alignment_path):
  """
  Make an empty diretory for saving automatic alignment results. 
  """
  if os.path.isdir(auto_alignment_path):
    shutil.rmtree(auto_alignment_path)
  os.makedirs(auto_alignment_path, exist_ok=True)
  
if __name__ == '__main__':
  t_0 = time.time()
  main()
  print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
