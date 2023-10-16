import os
from bertalign import Bertalign
from bertalign.eval import *

src_dir = 'text+berg/de'
tgt_dir = 'text+berg/fr'
output_dir = 'output_alignments'  # 新增：用于存储对齐结果的目录

# 创建存储对齐结果的目录
os.makedirs(output_dir, exist_ok=True)

test_alignments = []  # 定义空的列表来存储对齐结果
gold_alignments = []

for file in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file).replace("\\", "/")
    tgt_file = os.path.join(tgt_dir, file).replace("\\", "/")
    src = open(src_file, 'rt', encoding='utf-8').read()
    tgt = open(tgt_file, 'rt', encoding='utf-8').read()

    print("Start aligning {} to {}".format(src_file, tgt_file))
    aligner = Bertalign(src, tgt, is_split=True)
    aligner.align_sents()

    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"{file}.txt")

    # 将对齐结果写入文件
    aligner.write_sents_to_file(output_file)
    test_alignments.append(aligner.result)
