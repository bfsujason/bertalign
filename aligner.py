# -*- coding: utf-8 -*-


from bertalign import Bertalign
src = """
去年報載華視連續劇《包青天》受到觀眾熱烈歡迎，弟子們說：「師父就像是佛光山的包青天，常常及時伸出正義的援手，專門為大家排難解紛。」<br/>
設身處地\u3000謀求大家滿意<br/>
人間佛教<br/>
回憶自我懂事以來，就經常看到母親為鄰里親友排難解紛，記得曾經有人向他說：「何必多管閒事呢？」
母親聽了，正色答道：「排難解紛能促進別人的和諧美滿，是正事，怎麼能說是閒事呢？」
及至行腳台灣，先是落腳在佛寺中，搬柴、運水、拉車、採購……無所不做。<br/>
在耳濡目染下，我也繼承了母親的性格，一直都很喜歡幫助別人化解紛爭，而且並不一定是佛光山的徒眾，我才特意關懷照顧！<br/>
"""

tgt = """
A TV drama series depicting the life of Pao Ch’ing-t’ien (also known as Pao Cheng) was the most watched television show in Taiwan several years ago. 
Put Ourselves in Other People’s Places and Act on Their Behalf.
Humanistic Buddhism.
My disciples have often said about me, “Master is the Pao Ch’ing-t’ien of Fo Guang Shan because whenever there is a dispute, he promptly lends a hand and settles it justly.”  
I inherited my mother’s character. 
As far as I can remember, she served as mediator for quarreling neighbors and relatives. 
Someone once asked her, “Why must you meddle in others’ affairs?” 
“To settle conflicts,” my mother sternly replied, “is no trifling matter; it is a serious business because it promotes harmony and happiness in people’s lives.” 
Imbued with what I often saw, I have always taken great pleasure helping settle disputes. 
When later my wanderings in search of Buddhist teaching took me as far away as Taiwan, I first settled in a monastery, where I carried firewood, hauled water, pulled carts, made purchases, and patrolled the mountainscape night and day.
Nor are my care and concern limited to the disciples and followers of Fo Guang Shan. 
"""

aligner = Bertalign(src, tgt)
aligner.align_sents()
aligner.print_sents()

#output_file = "alignment_result.txt"  # 生成文件名,附带后缀，一般是txt，例“佛教概论.txt”
#aligner.write_sents_to_file(output_file)