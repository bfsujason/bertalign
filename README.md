# Bertalign
Word Embedding-Based Bilingual Sentence Aligner

## Evaluation Corpus
This section describes the procedure of creating the evaluation corpora: the manually aligned corpus (MAC) of Chinese-English literary texts and the Bible corpus aligned at the verse level.
### MAC-Test
The **MAC-Test** is saved in [corpus/mac/test](./corpus/mac/test)

The sampling scheme for building MAC-Test can be found at [meta_data.tsv](./corpus/mac/test/meta_data.tsv)

There are 4 subdirectories in MAC-Test.

The [split](./corpus/mac/test/split) directory contains the sentence-split source texts, target texts and the machine translations of source texts, which are required by *Bleualign* to perform automatic alignment.

The inputs to *Hunalign* are saved in the [tok](./corpus/mac/test/tok) directory.

The [emb](./corpus/mac/test/emb) directory is made up of the overlapping sentences and their embeddings for *Vecalign* and *BertAlign*.

We use [Intertext](https://wanthalf.saga.cz/intertext) to create the manual alignment for MAC and save the gold alignments in the [intertext](./corpus/mac/test/intertext) directory.

In order to facilitate system evaluations, we delete the XML tags and save the clean gold alignment file with only sentence IDs in the [gold](./eval/mac/test/gold) directory

### Bible
The **Bible** corpus is saved in [corpus/bible](./corpus/bible)

The directory makeup is similar to MAC-Test, except that there is no *intertext* directory for manual alignments.

The gold alignments for the Bible corpus are generated automatically from the original verse-aligned Bible corpus and saved in [eval/bible/gold](./eval/bible/gold)

In order to compare the sentence-based alignments returned by various aligners with the verse-based gold alignments, we put the verse ID for each sentence in the files *corpus/bible/en.verse* and *corpus/bible/zh.verse*, which are used to merge consecutive sentences in the output if they belong to the same verse.

## System Comparisons
All the experiments reported in the paper are conducted using [Google Colab](https://colab.research.google.com/)
### Job File
Before performing the automatic alignment, a job file is created for each aligner for batch processing. Each row in the job file represents an alignment task, which is made of three tab-separated file names for source, target and output text.

The job files for MAC-Test and Bible are located in *eval/mac/test/job* and *eval/bible/job* respectively.

### Sentence Embeddings
Before embedding the source and target sentences, we use the following Python script to create the combinations of consecutive sentences:
```
# MAC-Test
python utils/overlap.py -i corpus/mac/test/split -o corpus/mac/test/emb/en.overlap –l en –n 8
python utils/overlap.py -i corpus/mac/test/split -o corpus/mac/test/emb/zh.overlap –l zh –n 8

# Bible
python utils/overlap.py -i corpus/bible/split -o corpus/bible/en.overlap –l en –n 5
python utils/overlap.py -i corpus/bible/split -o corpus/bible/zh.overlap –l zh –n 5
```
Use parameters -i to specify the input data directory and -o the output file path.

All the file suffixes in the input directory should end with the corresponding language code, e.g. 001.en and 001.zh etc., and match up with the parameter -l.

The parameter -n indicates the number of overlapping sentences, which is similar to word n-grams applied to sentences.

We use [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) to convert texts into embeddings.

To install Sentence Transformers, just run:
```
pip install sentence-transformers
```
After the installation, we run the following Python script to embed the bitexts to be aligned:
```
# MAC-Test
python utils/embed.py –i corpus/mac/test/emb/en.overlap –o corpus/mac/test/emb/en.overlap.emb
python utils/embed.py –i corpus/mac/test/emb/zh.overlap –o corpus/mac/test/emb/zh.overlap.emb

# Bible
python utils/embed.py –i corpus/bible/emb/en.overlap –o corpus/bible/emb/en.overlap.emb
python utils/embed.py –i corpus/bible/emb/zh.overlap –o corpus/bible/emb/zh.overlap.emb
```
The parameter -i indicates the file containing sentence combinations.

We use the [tofile](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html) method provided by Python’s Numpy module to save the sentence embeddings in the file designated by -o.

### Evaluation on MAC-Test
#### Gale-Church
```
%timeit !python bin/gale_align.py --job eval/mac/test/job/galechurch.job

perl utils/eval_mac.pl --meta corpus/mac/test/meta_data.tsv -gold eval/mac/test/gold --auto eval/mac/test/auto/galechurch \
  --by book
```
#### Hunalign
```
%timeit !bin/hunalign/hunalign -text -batch bin/hunalign/ec.dic eval/mac/test/job/hunalign.job

perl utils/eval_mac.pl --meta corpus/mac/test/meta_data.tsv -gold eval/mac/test/gold --auto eval/mac/test/auto/hunalign \
  --by book
```
#### Bleualign
```
%timeit !python bin/bleualign/batch_align.py eval/mac/test/job/bleualign.job

perl utils/eval_mac.pl --meta corpus/mac/test/meta_data.tsv -gold eval/mac/test/gold --auto eval/mac/test/auto/bleualign \
  --by book
```
#### Vecalign
```
%timeit !python bin/vecalign/vecalign.py --job eval/mac/test/job/vecalign.job \
  --src_embed corpus/mac/test/emb/zh.overlap corpus/mac/test/emb/zh.overlap.emb \
  --tgt_embed corpus/mac/test/emb/en.overlap corpus/mac/test/emb/en.overlap.emb \
  --alignment_max_size 8
  
perl utils/eval_mac.pl --meta corpus/mac/test/meta_data.tsv -gold eval/mac/test/gold --auto eval/mac/test/auto/vecalign \
  --by book
```
#### Bertalign (Modified Cosine)
```
%timeit !python /bin/bert_align.py eval/mac/test/job/mbert.job \
  --src_embed corpus/mac/test/emb/zh.overlap corpus/mac/test/emb/zh.overlap.emb \
  --tgt_embed corpus/mac/test/emb/en.overlap corpus/mac/test/emb/en.overlap.emb \
  --margin --max_align 8
  
perl utils/eval_mac.pl --meta corpus/mac/test/meta_data.tsv -gold eval/mac/test/gold --auto eval/mac/test/auto/mbert \
  --by book
```
### Evaluation on Bible
#### Gale-Church
```
%timeit !python bin/gale_align.py --job eval/bible/job/galechurch.job

perl utils/eval_bible.pl --meta corpus/bible/meta_data.tsv --gold eval/bible/gold --auto eval/bible/auto/galechurch \
  --src_verse corpus/bible/en.verse --tgt_verse corpus/bible/zh.verse
```
#### Hunalign
```
%timeit !bin/hunalign/hunalign -text -batch bin/hunalign/ce.dic eval/bible/job/hunalign.job

perl utils/eval_bible.pl --meta corpus/bible/meta_data.tsv --gold eval/bible/gold --auto eval/bible/auto/hunalign \
  --src_verse corpus/bible/en.verse --tgt_verse corpus/bible/zh.verse
```
#### Bleualign (Run OOM on 25,000 sentences)
```
%timeit !python bin/bleualign/batch_align.py eval/bible/job/bleualign.job
```
#### Vecalign
```
%timeit !python bin/vecalign/vecalign.py --job eval/bible/job/vecalign.job \
  --src_embed corpus/bible/emb/en.overlap corpus/bible/emb/en.overlap.emb \
  --tgt_embed corpus/bible/emb/zh.overlap corpus/bible/emb/zh.overlap.emb
  
perl utils/eval_bible.pl --meta corpus/bible/meta_data.tsv --gold eval/bible/gold --auto eval/bible/auto/vecalign \
  --src_verse corpus/bible/en.verse --tgt_verse corpus/bible/zh.verse
```
#### Bertalign (Modified Cosine)
```
%timeit !python bin/bert_align.py --job eval/bible/job/mbert.job \
  --src_embed corpus/bible/emb/en.overlap corpus/bible/emb/en.overlap.emb \
  --tgt_embed corpus/bible/emb/zh.overlap corpus/bible/emb/zh.overlap.emb \
  --margin

perl utils/eval_bible.pl --meta corpus/bible/meta_data.tsv --gold eval/bible/gold --auto eval/bible/auto/mbert \
  --src_verse corpus/bible/en.verse --tgt_verse corpus/bible/zh.verse
```
