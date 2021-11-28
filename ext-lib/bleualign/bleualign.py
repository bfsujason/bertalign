# 2021/11/27
# bfsujason@163.com

"""
Usage:

python ext-lib/bleualign/bleualign.py \
  -m data/mac/test/meta_data.tsv \
  -s data/mac/test/zh \
  -t data/mac/test/en \
  -o data/mac/test/auto
"""

import os
import sys
import time
import shutil
import argparse

def main():
  parser = argparse.ArgumentParser(description='Sentence alignment using Bleualign')
  parser.add_argument('-s', '--src', type=str, required=True, help='Source directory.')
  parser.add_argument('-t', '--tgt', type=str, required=True, help='Target directory.')
  parser.add_argument('-o', '--out', type=str, required=True, help='Output directory.')
  parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file.')
  parser.add_argument('--tok', action='store_true', help='Use tokenized source trans and target text.')
  args = parser.parse_args()
  
  make_dir(args.out)
  
  jobs = create_jobs(args.meta, args.src, args.tgt, args.out, args.tok)
  job_path = os.path.abspath(os.path.join(args.out, 'bleualign.job'))
  write_jobs(jobs, job_path)
  
  bleualign_bin = os.path.abspath('ext-lib/bleualign/batch_align.py')
  run_bleualign(bleualign_bin, job_path)
  
  convert_format(args.out)

def convert_format(dir):
  for file in os.listdir(dir):
    if file.endswith('-s'):
      file_id = file.split('.')[0]
      src = os.path.join(dir, file)
      tgt = os.path.join(dir, file_id + '.align-t')
      out = os.path.join(dir, file_id + '.align')
      _convert_format(src, tgt, out)
      os.unlink(src)
      os.unlink(tgt)

def _convert_format(src, tgt, path):
  src_align = read_alignment(src)
  tgt_align = read_alignment(tgt)
  with open(path, 'wt', encoding='utf-8') as f:
    for x, y in zip(src_align, tgt_align):
      f.write("{}:{}\n".format(x,y))

def read_alignment(file):
  alignment = []
  with open(file, 'rt', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      alignment.append([int(x) for x in line.split(',')])
      
  return alignment
      
def run_bleualign(bin, job):
  cmd = "python {} {}".format(bin, job)
  os.system(cmd)
  os.unlink(job)
  
def write_jobs(jobs, path):
  jobs = '\n'.join(jobs)
  with open(path, 'wt', encoding='utf-8') as f:
    f.write(jobs)
       
def create_jobs(meta, src, tgt, out, is_tok):
  jobs = []
  fns = get_fns(meta)
  for file in fns:
    src_path = os.path.abspath(os.path.join(src, file))
    trans_path = os.path.abspath(os.path.join(src, file + '.trans'))
    if is_tok:
      tgt_path = os.path.abspath(os.path.join(tgt, file + '.tok'))
    else:
      tgt_path = os.path.abspath(os.path.join(tgt, file))
    out_path = os.path.abspath(os.path.join(out, file + '.align'))
    jobs.append('\t'.join([trans_path, src_path, tgt_path, out_path]))
    
  return jobs
  
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
  t_0 = time.time()
  main()
  print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
