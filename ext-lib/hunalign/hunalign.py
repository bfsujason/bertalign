# 2021/11/27
# bfsujason@163.com

"""
Usage:

python ext-lib/hunalign/hunalign.py \
  -m data/mac/test/meta_data.tsv \
  -s data/mac/test/zh \
  -t data/mac/test/en \
  -o data/mac/test/auto \
  -d ec.dic
"""

import os
import time
import shutil
import platform
import argparse

def main():
  parser = argparse.ArgumentParser(description='Sentence alignment using Hunalign')
  parser.add_argument('-s', '--src', type=str, required=True, help='Source directory.')
  parser.add_argument('-t', '--tgt', type=str, required=True, help='Target directory.')
  parser.add_argument('-o', '--out', type=str, required=True, help='Output directory.')
  parser.add_argument('-m', '--meta', type=str, required=True, help='Metadata file.')
  parser.add_argument('-d', '--dic', type=str, help='Dictionary file.')
  args = parser.parse_args()
  
  make_dir(args.out)
  
  jobs = create_jobs(args.meta, args.src, args.tgt, args.out)
  job_path = os.path.abspath(os.path.join(args.out, 'hunalign.job'))
  write_jobs(jobs, job_path)
  
  if args.dic:
    hunalign_dic = os.path.abspath(os.path.join('ext-lib/hunalign', args.dic))
  else:
    hunalign_dic = os.path.abspath('ext-lib/hunalign/null.dic')
  
  # check system OS
  OS = platform.system()
  if OS == 'Windows':
    hunalign_bin = os.path.abspath('ext-lib/hunalign/hunalign.exe')
  elif OS == 'Linux':
    hunalign_bin = os.path.abspath('ext-lib/hunalign/hunalign')
  print(hunalign_bin)
  print(hunalign_dic)
  print(job_path)
  run_hunalign(hunalign_bin, hunalign_dic, job_path)
  convert_format(args.out)
  
def convert_format(dir):
  for file in sorted(os.listdir(dir)):
    fp_in = os.path.join(dir, file)
    fp_out = os.path.join(dir, file + '.align')
    alignment = _convert_format(fp_in, fp_out)
    write_alignment(alignment, fp_out)
    os.unlink(fp_in)

def _convert_format(fp_in, fp_out):
  src_id = -1
  tgt_id = -1
  alignment = []
  
  with open(fp_in, 'rt', encoding='utf-8') as f:
    for line in f:
      line = line.strip(' \r\n')
      items = line.split('\t');
      if not items[0] and not items[1]:
        continue
      src_seg_len, src_seg_id = _parse_seg(items[0], src_id)
      tgt_seg_len, tgt_seg_id = _parse_seg(items[1], tgt_id)
      src_id += src_seg_len
      tgt_id += tgt_seg_len
      alignment.append((src_seg_id, tgt_seg_id))
  
  return alignment

def write_alignment(alignment, fp_out):
  with open(fp_out, 'wt', encoding='utf-8') as f:
    for id in alignment:
      f.write("{}:{}\n".format(id[0], id[1]))
  
def _parse_seg(seg, id):
  seg_len = 0
  seg_id = []
  if seg:
    sents = seg.split(' ~~~ ')
    seg_len = len(sents)
    seg_id = [id + x for x in range(1, seg_len+1)]
   
  return seg_len, seg_id

def run_hunalign(bin, dic, job):
  cmd = "{} -text -batch {} {}".format(bin, dic, job)
  os.system(cmd)
  os.unlink(job)
  
def write_jobs(jobs, path):
  jobs = '\n'.join(jobs)
  with open(path, 'wt', encoding='utf-8', newline='\n') as f:
    f.write(jobs)
   
def create_jobs(meta, src, tgt, out):
  jobs = []
  fns = get_fns(meta)
  for file in fns:
    # using tokenized file
    src_path = os.path.abspath(os.path.join(src, file + '.tok'))
    tgt_path = os.path.abspath(os.path.join(tgt, file + '.tok'))
    out_path = os.path.abspath(os.path.join(out, file))

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

def make_dir(path):
  if os.path.isdir(path):
    shutil.rmtree(path)
  os.makedirs(path, exist_ok=True)
  
if __name__ == '__main__':
  t_0 = time.time()
  main()
  print("It takes {:.3f} seconds to align all the sentences.".format(time.time() - t_0))
