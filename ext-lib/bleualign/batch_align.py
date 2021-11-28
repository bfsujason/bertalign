#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright: University of Zurich
# Author: Rico Sennrich

# script to allow batch-alignment of multiple files. No multiprocessing.
# syntax: python batch_align directory source_suffix target_suffix translation_suffix
#
# example: given the directory batch-test with the files 0.de, 0.fr and 0.trans, 1.de, 1.fr and 1.trans and so on,
# (0.trans being the translation of 0.de into the target language),
# then this command will align all files: python batch_align.py batch-test/ de fr trans
#
# output files will have ending source_suffix.aligned and target_suffix.aligned


import sys
import os
from bleualign.align import Aligner

if len(sys.argv) < 2:
    sys.stderr.write('Usage: python batch_align.py job_file\n')
    exit()

job_fn = sys.argv[1]

options = {}
options['factored'] = False
options['filter'] = None
options['filterthreshold'] = 90
options['filterlang'] = None
options['targettosrc'] = []
options['eval'] = None
options['galechurch'] = None
options['verbosity'] = 1
options['printempty'] = False

jobs = []
with open(job_fn, 'r', encoding="utf-8") as f:
    for line in f:
        if not line.startswith("#"):
            jobs.append(line.strip())

for rec in jobs:
    translation_document, source_document, target_document, out_document = rec.split("\t")
    options['srcfile'] = source_document
    options['targetfile'] = target_document
    options['srctotarget'] = [translation_document]
    options['output'] = out_document
    a = Aligner(options)
    a.mainloop()
    