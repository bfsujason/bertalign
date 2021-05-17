# Bertalign
word embedding-based bilingual sentence aligner

## Evaluation Corpus
This section describes the procedure of creating the evaluation corpora: the manually aligned corpus (MAC) of Chinese-English literary texts and the Bible corpus aligned at the verse level.
### MAC
Firstly, 5 chapters and their translations are sampled from each of the 6 novels included in MAC, obtaining a corpus of 30 bitexts. We then split the corpus into **MAC-Dev** and **MAC-Test** with the former containing 6 chapters and the latter 24 chapters.

The **MAC-Test** is saved in [corpus/mac/test](./corpus/mac/test)

The sampling schemes for building **MAC-Test** can be found at [corpus/mac/test/meta_data.tsv](./corpus/mac/test/meta_data.tsv)

There are 4 subdirectories in **MAC-Test**. The [split](/corpus/mac/test/split) directory contains the sentence-split source texts, target texts and the machine translations of source texts, which are required by **Bleualign** to perform automatic alignment.

The inputs to **Hunalign** are saved in the [tok](/corpus/mac/test/tok) directory.

The emb directory is made up of the overlapping sentences and their embeddings for Vecalign and BertAlign.
