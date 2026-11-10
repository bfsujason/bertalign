# Bertalign

An automatic mulitlingual sentence aligner.

Bertalign is designed to facilitate the construction of multilingual parallel corpora and translation memories, which have a wide range of applications in translation-related research such as corpus-based translation studies, contrastive linguistics, computer-assisted translation, translator education and machine translation.

## Approach

Bertalign uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) to represent source and target sentences so that semantically similar sentences in different languages are mapped onto similar vector spaces. Then a two-step algorithm based on dynamic programming is performed: 1) Step 1 finds the 1-1 alignments for approximate anchor points; 2) Step 2 limits the search path to the anchor points and extracts all the valid alignments with 1-many, many-1 or many-to-many relations between the source and target sentences.

## Performance

According to our experiments, Bertalign achieves more accurate results on [Text+Berg](./text+berg), a publicly available German-French parallel corpus, than the traditional length-, dictionary-, or MT-based alignment methods as reported in [Thompson & Koehn (2019)](https://aclanthology.org/D19-1136/)

## Languges Supported

Alignment between 25 languages: Catalan (ca), Chinese (zh), Czech (cs), Danish (da), Dutch (nl), English(en), Finnish (fi), French (fr), German (de), Greek (el), Hungarian (hu), Icelandic (is), Italian (it), Lithuanian (lt), Latvain (lv), Norwegian (no), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), and Turkish (tr).

## Installation

Please see [requirements.txt](./requirements.txt) for installation. 

### You can also install Bertalign and run the examples directly in a [Google Colab notebook](https://colab.research.google.com/drive/123GhXwgwmQp1F5SVZ74_uIgyxo6hLRq0?usp=sharing).

## Basic example

Just import *Bertalign* and initialize it with the source and target text, which will detect the source and target language automatically and split both texts into sentences. Then invoke the method *align_sents()*  to align sentences and print out the result with *print_sents()*.

```python
from bertalign import Bertalign
```

```python
src = """两年以后，大兴安岭。
“顺山倒咧——”
随着这声嘹亮的号子，一棵如巴特农神庙的巨柱般高大的落叶松轰然倒下，叶文洁感到大地抖动了一下。她拿起斧头和短锯，开始去除巨大树身上的枝丫。每到这时，她总觉得自己是在为一个巨人整理遗体。她甚至常常有这样的想象：这巨人就是自己的父亲。两年前那个凄惨的夜晚，她在太平间为父亲整理遗容时的感觉就在这时重现。巨松上那绽开的树皮，似乎就是父亲躯体上累累的伤痕。
内蒙古生产建设兵团的六个师四十一个团十多万人就分布在这辽阔的森林和草原之间。刚从城市来到这陌生的世界时，很多兵团知青都怀着一个浪漫的期望：当苏修帝国主义的坦克集群越过中蒙边境时，他们将飞快地武装起来，用自己的血肉构成共和国的第一道屏障。事实上，这也确实是兵团组建时的战略考虑之一。但他们渴望的战争就像草原天边那跑死马的远山，清晰可见，但到不了眼前，于是他们只有垦荒、放牧和砍伐。这些曾在“大串联”中燃烧青春的年轻人很快发现，与这广阔天地相比，内地最大的城市不过是个羊圈；在这寒冷无际的草原和森林间，燃烧是无意义的，一腔热血喷出来，比一堆牛粪凉得更快，还不如后者有使用价值。但燃烧是他们的命运，他们是燃烧的一代。于是，在他们的油锯和电锯下，大片的林海化为荒山秃岭；在他们的拖拉机和康拜因（联合收割机）下，大片的草原被犁成粮田，然后变成沙漠。
叶文洁看到的砍伐只能用疯狂来形容，高大挺拔的兴安岭落叶松、四季常青的樟子松、亭亭玉立的白桦、耸入云天的山杨、西伯利亚冷杉，以及黑桦、柞树、山榆、水曲柳、钻天柳、蒙古栎，见什么伐什么，几百把油锯如同一群钢铁蝗虫，她的连队所过之处，只剩下一片树桩。
整理好的落叶松就要被履带拖拉机拖走了，在树干另一头，叶文洁轻轻抚摸了一下那崭新的锯断面，她常常下意识地这么做，总觉得那是一处巨大的伤口，似乎能感到大树的剧痛。她突然看到，在不远处树桩的锯断面上，也有一只在轻轻抚摸的手，那手传达出的心灵的颤抖，与她产生了共振。那手虽然很白皙，但能够看出是属于男性的。叶文洁抬头，看到抚摸树桩的人是白沐霖，一个戴眼镜的瘦弱青年，他是兵团《大生产报》的记者，前天刚到连队来采访。叶文洁看过他写的文章，文笔很好，其中有一种与这个粗放环境很不协调的纤细和敏感，令她很难忘。"""

tgt = """Two years later, the Greater Khingan Mountains
“Tim-ber…”
Following the loud chant, a large Dahurian larch, thick as the columns of the Parthenon, fell with a thump, and Ye Wenjie felt the earth quake.
She picked up her ax and saw and began to clear the branches from the trunk. Every time she did this, she felt as though she were cleaning the corpse of a giant. Sometimes she even imagined the giant was her father. The feelings from that terrible night two years ago when she cleaned her father’s body in the mortuary would resurface, and the splits and cracks in the larch bark seemed to turn into the old scars and new wounds covering her father.
Over one hundred thousand people from the six divisions and forty-one regiments of the Inner Mongolia Production and Construction Corps were scattered among the vast forests and grasslands. When they first left the cities and arrived at this unfamiliar wilderness, many of the corps’ “educated youths”—young college students who no longer had schools to go to—had cherished a romantic wish: When the tank clusters of the Soviet Revisionist Imperialists rolled over the Sino-Mongolian border, they would arm themselves and make their own bodies the first barrier in the Republic’s defense. Indeed, this expectation was one of the strategic considerations motivating the creation of the Production and Construction Corps.
But the war they craved was like a mountain at the other end of the grassland: clearly visible, but as far away as a mirage. So they had to content themselves with clearing fields, grazing animals, and chopping down trees.
Soon, the young men and women who had once expended their youthful energy on tours to the holy sites of the Chinese Revolution discovered that, compared to the huge sky and open air of Inner Mongolia, the biggest cities in China’s interior were nothing more than sheep pens. Stuck in the middle of the cold, endless expanse of forests and grasslands, their burning ardor was meaningless. Even if they spilled all of their blood, it would cool faster than a pile of cow dung, and not be as useful. But burning was their fate; they were the generation meant to be consumed by fire. And so, under their chain saws, vast seas of forests turned into barren ridges and denuded hills. Under their tractors and combine harvesters, vast tracts of grasslands became grain fields, then deserts.
Ye Wenjie could only describe the deforestation that she witnessed as madness. The tall Dahurian larch, the evergreen Scots pine, the slim and straight white birch, the cloud-piercing Korean aspen, the aromatic Siberian fir, along with black birch, oak, mountain elm, Chosenia arbutifolia—whatever they laid eyes on, they cut down. Her company wielded hundreds of chain saws like a swarm of steel locusts, and after they passed, only stumps were left.
The fallen Dahurian larch, now bereft of branches, was ready to be taken away by tractor. Ye gently caressed the freshly exposed cross section of the felled trunk. She did this often, as though such surfaces were giant wounds, as though she could feel the tree’s pain. Suddenly, she saw another hand lightly stroking the matching surface of the stump a few feet away. The tremors in that hand revealed a heart that resonated with hers. Though the hand was pale, she could tell it belonged to a man.
She looked up. It was Bai Mulin. A slender, delicate man who wore glasses, he was a reporter for the Great Production News, the corps’ newspaper. He had arrived the day before yesterday to gather news about her company. Ye remembered reading his articles, which were written in a beautiful style, sensitive and fine, ill suited to the rough-hewn environment."""
```

```python
aligner = Bertalign(src, tgt)
aligner.align_sents()
```

    Source language: Chinese, Number of sentences: 21
    Target language: English, Number of sentences: 32
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 21 Chinese sentences to 32 English sentences

```python
aligner.print_sents()
```

    两年以后，大兴安岭。
    Two years later, the Greater Khingan Mountains
    
    “顺山倒咧——”
    “Tim-ber…”
    
    随着这声嘹亮的号子，一棵如巴特农神庙的巨柱般高大的落叶松轰然倒下，叶文洁感到大地抖动了一下。
    Following the loud chant, a large Dahurian larch, thick as the columns of the Parthenon, fell with a thump, and Ye Wenjie felt the earth quake.
    
    她拿起斧头和短锯，开始去除巨大树身上的枝丫。
    She picked up her ax and saw and began to clear the branches from the trunk.
    
    每到这时，她总觉得自己是在为一个巨人整理遗体。
    Every time she did this, she felt as though she were cleaning the corpse of a giant.
    
    她甚至常常有这样的想象：这巨人就是自己的父亲。
    Sometimes she even imagined the giant was her father.
    
    两年前那个凄惨的夜晚，她在太平间为父亲整理遗容时的感觉就在这时重现。 巨松上那绽开的树皮，似乎就是父亲躯体上累累的伤痕。
    The feelings from that terrible night two years ago when she cleaned her father’s body in the mortuary would resurface, and the splits and cracks in the larch bark seemed to turn into the old scars and new wounds covering her father.
    
    内蒙古生产建设兵团的六个师四十一个团十多万人就分布在这辽阔的森林和草原之间。
    Over one hundred thousand people from the six divisions and forty-one regiments of the Inner Mongolia Production and Construction Corps were scattered among the vast forests and grasslands.
    
    刚从城市来到这陌生的世界时，很多兵团知青都怀着一个浪漫的期望：当苏修帝国主义的坦克集群越过中蒙边境时，他们将飞快地武装起来，用自己的血肉构成共和国的第一道屏障。
    When they first left the cities and arrived at this unfamiliar wilderness, many of the corps’ “educated youths”—young college students who no longer had schools to go to—had cherished a romantic wish: When the tank clusters of the Soviet Revisionist Imperialists rolled over the Sino-Mongolian border, they would arm themselves and make their own bodies the first barrier in the Republic’s defense.
    
    事实上，这也确实是兵团组建时的战略考虑之一。
    Indeed, this expectation was one of the strategic considerations motivating the creation of the Production and Construction Corps.
    
    但他们渴望的战争就像草原天边那跑死马的远山，清晰可见，但到不了眼前，于是他们只有垦荒、放牧和砍伐。
    But the war they craved was like a mountain at the other end of the grassland: clearly visible, but as far away as a mirage. So they had to content themselves with clearing fields, grazing animals, and chopping down trees.
    
    这些曾在“大串联”中燃烧青春的年轻人很快发现，与这广阔天地相比，内地最大的城市不过是个羊圈；在这寒冷无际的草原和森林间，燃烧是无意义的，一腔热血喷出来，比一堆牛粪凉得更快，还不如后者有使用价值。
    Soon, the young men and women who had once expended their youthful energy on tours to the holy sites of the Chinese Revolution discovered that, compared to the huge sky and open air of Inner Mongolia, the biggest cities in China’s interior were nothing more than sheep pens. Stuck in the middle of the cold, endless expanse of forests and grasslands, their burning ardor was meaningless. Even if they spilled all of their blood, it would cool faster than a pile of cow dung, and not be as useful.
    
    但燃烧是他们的命运，他们是燃烧的一代。
    But burning was their fate; they were the generation meant to be consumed by fire.
    
    于是，在他们的油锯和电锯下，大片的林海化为荒山秃岭；在他们的拖拉机和康拜因（联合收割机）下，大片的草原被犁成粮田，然后变成沙漠。
    And so, under their chain saws, vast seas of forests turned into barren ridges and denuded hills. Under their tractors and combine harvesters, vast tracts of grasslands became grain fields, then deserts.
    
    叶文洁看到的砍伐只能用疯狂来形容，高大挺拔的兴安岭落叶松、四季常青的樟子松、亭亭玉立的白桦、耸入云天的山杨、西伯利亚冷杉，以及黑桦、柞树、山榆、水曲柳、钻天柳、蒙古栎，见什么伐什么，几百把油锯如同一群钢铁蝗虫，她的连队所过之处，只剩下一片树桩。
    Ye Wenjie could only describe the deforestation that she witnessed as madness. The tall Dahurian larch, the evergreen Scots pine, the slim and straight white birch, the cloud-piercing Korean aspen, the aromatic Siberian fir, along with black birch, oak, mountain elm, Chosenia arbutifolia—whatever they laid eyes on, they cut down. Her company wielded hundreds of chain saws like a swarm of steel locusts, and after they passed, only stumps were left.
    
    整理好的落叶松就要被履带拖拉机拖走了，在树干另一头，叶文洁轻轻抚摸了一下那崭新的锯断面，她常常下意识地这么做，总觉得那是一处巨大的伤口，似乎能感到大树的剧痛。
    The fallen Dahurian larch, now bereft of branches, was ready to be taken away by tractor. Ye gently caressed the freshly exposed cross section of the felled trunk. She did this often, as though such surfaces were giant wounds, as though she could feel the tree’s pain.
    
    她突然看到，在不远处树桩的锯断面上，也有一只在轻轻抚摸的手，那手传达出的心灵的颤抖，与她产生了共振。
    Suddenly, she saw another hand lightly stroking the matching surface of the stump a few feet away. The tremors in that hand revealed a heart that resonated with hers.
    
    那手虽然很白皙，但能够看出是属于男性的。
    Though the hand was pale, she could tell it belonged to a man.
    
    叶文洁抬头，看到抚摸树桩的人是白沐霖，一个戴眼镜的瘦弱青年，他是兵团《大生产报》的记者，前天刚到连队来采访。
    She looked up. It was Bai Mulin. A slender, delicate man who wore glasses, he was a reporter for the Great Production News, the corps’ newspaper. He had arrived the day before yesterday to gather news about her company.
    
    叶文洁看过他写的文章，文笔很好，其中有一种与这个粗放环境很不协调的纤细和敏感，令她很难忘。
    Ye remembered reading his articles, which were written in a beautiful style, sensitive and fine, ill suited to the rough-hewn environment.

## Batch processing & evaluation

The following example shows how to use Bertalign to align the Text+Berg corpus, and evaluate its performance with gold standard alignments. The evaluation script [eval.py](./bertalign/eval.py) is based on [Vecalign](https://github.com/thompsonb/vecalign).

Please see [aligner.py](./bertalign/aligner.py) for more options to configure Bertalign.

```python
import os
from bertalign import Bertalign
from bertalign.eval import * 
```

```python
src_dir = 'text+berg/de'
tgt_dir = 'text+berg/fr'
gold_dir = 'text+berg/gold'
```

```python
test_alignments = []
gold_alignments = []
for file in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file).replace("\\","/")
    tgt_file = os.path.join(tgt_dir, file).replace("\\","/")
    src = open(src_file, 'rt', encoding='utf-8').read()
    tgt = open(tgt_file, 'rt', encoding='utf-8').read()

    print("Start aligning {} to {}".format(src_file, tgt_file))
    aligner = Bertalign(src, tgt, is_split=True)
    aligner.align_sents()
    test_alignments.append(aligner.result)

    gold_file = os.path.join(gold_dir, file)
    gold_alignments.append(read_alignments(gold_file))
```

    Start aligning text+berg/de/001 to text+berg/fr/001
    Source language: German, Number of sentences: 137
    Target language: French, Number of sentences: 155
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 137 German sentences to 155 French sentences
    
    Start aligning text+berg/de/002 to text+berg/fr/002
    Source language: German, Number of sentences: 293
    Target language: French, Number of sentences: 274
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 293 German sentences to 274 French sentences
    
    Start aligning text+berg/de/003 to text+berg/fr/003
    Source language: German, Number of sentences: 95
    Target language: French, Number of sentences: 100
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 95 German sentences to 100 French sentences
    
    Start aligning text+berg/de/004 to text+berg/fr/004
    Source language: German, Number of sentences: 107
    Target language: French, Number of sentences: 112
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 107 German sentences to 112 French sentences
    
    Start aligning text+berg/de/005 to text+berg/fr/005
    Source language: German, Number of sentences: 36
    Target language: French, Number of sentences: 40
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 36 German sentences to 40 French sentences
    
    Start aligning text+berg/de/006 to text+berg/fr/006
    Source language: German, Number of sentences: 126
    Target language: French, Number of sentences: 131
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 126 German sentences to 131 French sentences
    
    Start aligning text+berg/de/007 to text+berg/fr/007
    Source language: German, Number of sentences: 197
    Target language: French, Number of sentences: 199
    Embedding source and target text using LaBSE ...
    Performing first-step alignment ...
    Performing second-step alignment ...
    Finished! Successfuly aligning 197 German sentences to 199 French sentences

```python
scores = score_multiple(gold_list=gold_alignments, test_list=test_alignments)
log_final_scores(scores)
```

     ---------------------------------
    |             |  Strict |    Lax  |
    | Precision   |   0.932 |   0.987 |
    | Recall      |   0.941 |   0.991 |
    | F1          |   0.936 |   0.989 |
     ---------------------------------

## Citation

Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, *Digital Scholarship in the Humanities*. [https://doi.org/10.1093/llc/fqac089](https://doi.org/10.1093/llc/fqac089).

## Funding

The work is supported by the MOE Foundation of Humanities and Social Sciences (Grant No. 17YJC740055).

## Licence

Bertalign is released under the [GNU General Public License v3.0](./LICENCE)

## Credits

##### Main Libraries

* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

* [faiss](https://github.com/facebookresearch/faiss)

* [sentence-splitter](https://github.com/mediacloud/sentence-splitter)

##### Other Sentence Aligners

* [Hunalign](http://mokk.bme.hu/en/resources/hunalign/)

* [Bleualign](https://github.com/rsennrich/Bleualign)

* [Vecalign](https://github.com/thompsonb/vecalign)

## Todo List

- Try the [CNN model](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) for sentence embeddings
* Develop a GUI for Windows users
