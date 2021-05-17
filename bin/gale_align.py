import os
import sys
import argparse
import time
import math
import numba as nb
import numpy as np

def _main():
  # user-defined parameters
  parser = argparse.ArgumentParser('Sentence alignment using Gale-Church Algrorithm',
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--job', type=str, required=True, help='Job file for alignment task.')
  args = parser.parse_args()
  
  # fixed parameters to determine the
  # window size for alignment  
  min_win_size = 10
  max_win_size = 600
  win_per_100 = 8
  
  # alignment types
  align_types = np.array([
    [0,1],
    [1,0],
    [1,1],
    [1,2],
    [2,1],
    [2,2]
  ], dtype=np.int)
  
  # prior probability
  priors = np.array([0, 0.0099, 0.89, 0.089, 0.011])
  
  # mean and variance
  c = 1
  s2 = 6.8
  
  # gale church align
  job = read_job(args.job)
  for rec in job:
    src_file, tgt_file, align_file = rec.split("\t")
    print("Aligning {} to {}".format(src_file, tgt_file))
    src_lines = open(src_file, 'rt', encoding="utf-8").readlines() # UTF-8 byte length
    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
    src_len = calculate_txt_len(src_lines)
    tgt_len = calculate_txt_len(tgt_lines)
    
    m = src_len.shape[0] - 1
    n = tgt_len.shape[0] - 1
  
    # find search path
    w, search_path = \
      find_search_path(m, n, min_win_size, max_win_size, win_per_100)
    
    cost, back = align(src_len, tgt_len, w, search_path, align_types, priors, c, s2)
    alignment = back_track(m, n, back, search_path, align_types)
    #print(alignment)
    
    # save alignment
    f = open(align_file, 'w', encoding="utf-8")
    print_alignments(alignment, file=f)
    
def print_alignments(alignments, file=sys.stdout):
  for x, y in alignments:
    print('%s:%s' % (x, y), file=file)
    
def back_track(i, j, b, search_path, a_types):
    #i = b.shape[0] - 1
    #j = b.shape[1] - 1
    alignment = []
    while ( i !=0  and j != 0 ):
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
def align(src_len, tgt_len, w, search_path, align_types, priors, c, s2):

  #initialize cost and backpointer matrix
  m = src_len.shape[0] - 1
  cost = np.zeros((m + 1, 2 * w + 1))
  back = np.zeros((m + 1, 2 * w + 1), dtype=nb.int64)
  cost[0][0] = 0
  back[0][0] = -1

  for i in range(m + 1):
    i_start = search_path[i][0]
    i_end = search_path[i][1]

    for j in range(i_start, i_end + 1):
      if i + j == 0:
        continue
   
      best_score = np.inf
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

        score = cost[prev_i][prev_j_offset] - math.log(priors[a_1 + a_2]) + \
          get_score(src_len[i] - src_len[i - a_1], tgt_len[j] - tgt_len[j - a_2], c, s2)
        
        if score < best_score:
          best_score = score
          best_a = a
      
      j_offset = j - i_start
      cost[i][j_offset] = best_score
      back[i][j_offset] = best_a
   
  return cost, back
  
@nb.jit(nopython=True, fastmath=True, cache=True)
def get_score(len_s, len_t, c, s2):
  
  mean = (len_s + len_t / c) / 2
  z = (len_t - len_s * c) / math.sqrt(mean * s2)
  
  pd = 2 * (1 - norm_cdf(abs(z)))
  if pd > 0:
    return -math.log(pd)
    
  return 25
  
@nb.jit(nopython=True, fastmath=True, cache=True)
def find_search_path(src_len, tgt_len, min_win_size, max_win_size, win_per_100):
  yx_ratio = tgt_len / src_len
  win_size_1 = int(yx_ratio * tgt_len * win_per_100 / 100)
  win_size_2 = int(abs(tgt_len - src_len) * 3/4)
  w_1 = min(max(min_win_size, max(win_size_1, win_size_2)), max_win_size)
  #w_2 = int(max(src_len, tgt_len) * 0.05)
  w_2 = int(max(src_len, tgt_len) * 0.06)
  w = max(w_1, w_2)
  search_path = np.zeros((src_len + 1, 2), dtype=nb.int64)
  for i in range(0, src_len + 1):
    center = int(yx_ratio * i)
    w_start = max(0, center - w)
    w_end = min(center + w, tgt_len)
    search_path[i] = [w_start, w_end]
  return w, search_path
  
@nb.jit(nopython=True, fastmath=True, cache=True)
def norm_cdf(z):
  t = 1/float(1+0.2316419*z) # t = 1/(1+pz) , z=0.2316419
  p_norm = 1 - 0.3989423*math.exp(-z*z/2) * ((0.319381530 * t)+ \
                                         (-0.356563782 * t)+ \
                                         (1.781477937 * t) + \
                                         (-1.821255978* t) + \
                                         (1.330274429 * t))
  return p_norm
  
def calculate_txt_len(lines):
    txt_len = []
    txt_len.append(0)
    for i, line in enumerate(lines):
        txt_len.append(txt_len[i] + len(line.strip().encode("utf-8")))
    return np.array(txt_len)

def read_job(file):
    job = []
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                job.append(line.strip())
    return job

if __name__ == '__main__':
    t_0 = time.time()
    _main()
    print("It takes {}".format(time.time() - t_0))