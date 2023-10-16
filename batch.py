import os
from bertalign import Bertalign
from bertalign.eval import *

src_dir = 'text+berg/Chinese'           # 原始文件路径
tgt_dir = 'text+berg/English'           # 译文文件路径 注：中英文对照文件名字要保持一致
output_dir = 'text+berg/output_alignments'  # 新增：用于存储对齐结果的目录

# 创建存储对齐结果的目录
os.makedirs(output_dir, exist_ok=True)

test_alignments = []  # 定义空的列表来存储对齐结果

for file in os.listdir(src_dir):
    src_file = os.path.join(src_dir, file).replace("\\", "/")
    tgt_file = os.path.join(tgt_dir, file).replace("\\", "/")

    if os.path.isdir(src_file) or os.path.isdir(tgt_file):
        continue

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
