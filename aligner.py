# -*- coding: utf-8 -*-

from bertalign import Bertalign
src = """
事实上，与其你担心妄语，你可以正语啊。
因为那位信徒是开布店，有人要买布了，就会问他：「布多少钱一尺？」「五块钱。」「褪色不褪色？」为了要能卖出去，他只有说谎：「不褪色。」
后来我告诉他，你不要这样说，你可以说：「五块钱的会褪色，另外有个八块钱的不褪色。」
后来这位信徒盖了大楼，就是因为他的诚信，不妄语，让他生意兴隆。我只是觉得奇怪，这么好的佛法，为什么不把它作为积极的解说，让信徒受到佛法的利益呢？
"""

tgt = """
Instead of worrying about lying, one can choose to practice  Right Speech. Once there was a Buddhist who owned a textile store,  and when customers wanted to buy a piece of cloth, they would ask,  “How much for a foot of fabric?”
“Five dollars.”
“Does the color fade?”
To sell the cloth, he would lie and say, “No, it doesn’t.” Later, I told him not to respond that way. 
Instead, I suggested  that he could say, “The five-dollar fabric fades easily, but the eight dollar one does not.” 
Due to the honorable reputation he gained from  being honest, his business boomed, which allowed him to build an  establishment. 
Dharma brings goodness to all. My only concern is,  why do people not positively explain Buddhism so devotees can  receive the benefit of the Dharma?
"""

aligner = Bertalign(src, tgt)
aligner.align_sents()

output_file = "alignment_result"
aligner.write_sents_to_file(output_file)