#!/usr/bin/env python3

import os
import argparse

def go(output_file, input_dir, num_overlaps, lang):
    output = set()
    for fin in os.listdir(input_dir):
        if fin.endswith('.' + lang):
            fpath = os.path.join(input_dir, fin)
            lines = open(fpath, 'rt', encoding="utf-8").readlines()
            for out_line in yield_overlaps(lines, num_overlaps):
                output.add(out_line)

    # for reproducibility
    output = list(output)
    output.sort()

    with open(output_file, 'wt', encoding="utf-8") as fout:
        for line in output:
            fout.write(line + '\n')

def yield_overlaps(lines, num_overlaps):
    lines = [preprocess_line(line) for line in lines]
    for overlap in range(1, num_overlaps + 1):
        for out_line in layer(lines, overlap):
            # check must be here so all outputs are unique
            out_line2 = out_line[:10000]  # limit line so dont encode arbitrarily long sentences
            yield out_line2
            
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
    
def preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line
    
def _main():
    parser = argparse.ArgumentParser('Create text file containing overlapping sentences.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', type=str,
                        help='input directory.')
    parser.add_argument('-o', '--output', type=str,
                        help='output text file containing overlapping sentneces')
    parser.add_argument('-l', '--language', type=str,
                        help='language code')
    parser.add_argument('-n', '--num_overlaps', type=int, default=4,
                        help='Maximum number of allowed overlaps.')

    args = parser.parse_args()
    go(output_file=args.output,
       input_dir=args.input,
       num_overlaps=args.num_overlaps,
       lang=args.language)

if __name__ == '__main__':
    _main()
