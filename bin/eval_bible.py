import os
import argparse
from ast import literal_eval
from collections import defaultdict
from eval import score_multiple, log_final_scores

def main():
  parser = argparse.ArgumentParser('Evaluate aligment quality for Bible corpus')
  parser.add_argument('-t', '--test', type=str, required=True, help='Test alignment file.')
  parser.add_argument('-g', '--gold', type=str, required=True, help='Gold alignment file.')
  parser.add_argument('--src_verse', type=str, required=True, help='Source verse file.')
  parser.add_argument('--tgt_verse', type=str, required=True, help='Target verse file.')
  args = parser.parse_args()
  
  test_alignments = read_alignments(args.test)
  gold_alignments = read_alignments(args.gold)
  
  src_verse = get_verse(args.src_verse)
  tgt_verse = get_verse(args.tgt_verse)
  
  merged_test_alignments = merge_test_alignments(test_alignments, src_verse, tgt_verse)
  res = score_multiple(gold_list=[gold_alignments], test_list=[merged_test_alignments])
  log_final_scores(res)
  
def merge_test_alignments(alignments, src_verse, tgt_verse):
  merged_align = []
  last_beads_type = None
  
  for beads in alignments:
    beads_type = find_beads_type(beads, src_verse, tgt_verse)
    if not last_beads_type:
      merged_align.append(beads)
    else:
      if beads_type == last_beads_type:
        merged_align[-1][0].extend(beads[0])
        merged_align[-1][1].extend(beads[1])
      else:
        merged_align.append(beads)
        
    last_beads_type = beads_type
  
  return merged_align

def find_beads_type(beads, src_verse, tgt_verse):
  src_bead = beads[0]
  tgt_bead = beads[1]
    
  src_bead_type = find_bead_type(src_bead, src_verse)
  tgt_bead_type = find_bead_type(tgt_bead, tgt_verse)
  
  src_bead_len = len(src_bead_type)
  tgt_bead_len = len(tgt_bead_type)
  
  if src_bead_len != 1 or tgt_bead_len != 1:
    return None
  else:
    src_verse = src_bead_type[0]
    tgt_verse = tgt_bead_type[0]
    if src_verse != tgt_verse:
      if src_verse == 'NULL':
        return tgt_verse
      elif tgt_verse == 'NULL':
        return src_verse
      else:
        return None
    else:
      return src_verse

def find_bead_type(bead, verse):
  bead_type = ['NULL']
  if len(bead) > 0:
    bead_type = unique_list([verse[id] for id in bead])
  
  return bead_type

def unique_list(list):
  unique_list = []
  for x in list:
    if x not in unique_list:
      unique_list.append(x)

  return unique_list
  
def get_verse(file):
  verse = defaultdict()
  with open(file, 'rt', encoding='utf-8') as f:
    for (i, line) in enumerate(f):
     verse[i] = line.strip()

  return verse
    
def read_alignments(fin):
  alignments = []
  with open(fin, 'rt', encoding="utf-8") as infile:
    for line in infile:
      fields = [x.strip() for x in line.split(':') if len(x.strip())]
      if len(fields) < 2:
        raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
      try:
        src = literal_eval(fields[0])
        tgt = literal_eval(fields[1])
      except:
        raise Exception('Failed to parse line "%s"' % line.strip())
      alignments.append((src, tgt))

  return alignments
  
if __name__ == '__main__':
  main()
