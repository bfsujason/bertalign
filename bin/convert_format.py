# 2021/11/30
# bfsujason@163.com

# Usage:
# python -p xi -s ..\data\split\zh zh -t ..\data\split\en en -a ..\data\zh-en -f intertext

import os
import re
import shutil
import argparse
from ast import literal_eval

def main():
    parser = argparse.ArgumentParser(description='winVecAlign: VecAlign for Windows OS')
    parser.add_argument('-p', '--prj', type=str, required=True, help='Project name.')
    parser.add_argument('-s', '--src', type=str, nargs=2, required=True, help='Source directory and language code.')
    parser.add_argument('-t', '--tgt', type=str, nargs=2, required=True, help='Target directory and language code.')
    parser.add_argument('-a', '--alignment', type=str, required=True, help='Auomatic alignment directory.')
    parser.add_argument('-f', '--format', type=str, required=True, help='Output format.')
    args = parser.parse_args()
    
    out_dir = os.path.join(args.alignment, args.format)
    make_dir(out_dir)
    
    for file in os.listdir(args.alignment):
        if not file.endswith('.align'):
            continue
        file_id = file.split('.')[0]
        src_file = os.path.join(args.src[0], file_id)
        tgt_file = os.path.join(args.tgt[0], file_id)
        
        src_lines = read_lines(src_file)
        tgt_lines = read_lines(tgt_file)
        
        links = read_alignments(os.path.join(args.alignment, file))
        
        if args.format == 'intertext':
            src_name = "{}_{}".format(file_id, args.src[1])
            tgt_name = "{}_{}".format(file_id, args.tgt[1])
        
            toDoc = '.'.join([args.prj, tgt_name, 'xml'])
            fromDoc = '.'.join([args.prj, src_name, 'xml'])
            linkDoc = '.'.join([args.prj, src_name, tgt_name, 'xml'])
        
            write_sent_xml(src_lines, out_dir, fromDoc)
            write_sent_xml(tgt_lines, out_dir, toDoc)
            write_link_xml(links, out_dir, toDoc, fromDoc, linkDoc)
        elif args.format == 'tsv':
            tsvDoc = os.path.join(out_dir, args.prj + '.' + file + '.txt')
            write_tsv(src_lines, tgt_lines, links, tsvDoc)
        elif args.format == 'tmx':
            tmxDoc = os.path.join(out_dir, args.prj + '.' + file + '.tmx')
            write_tmx(src_lines, args.src[1], tgt_lines, args.tgt[1], links, tmxDoc)
        
def write_tsv(src_lines, tgt_lines, links, tsvDoc):
    tsv = []
    for bead in (links):
        src_line = get_line(bead[0], src_lines)
        tgt_line = get_line(bead[1], tgt_lines)
        tsv.append(src_line + "\t" + tgt_line)
    with open(tsvDoc, 'wt', encoding="utf-8") as f:
        f.write("\n".join(tsv))

def write_link_xml(links, dir, toDoc, fromDoc, linkDoc):
    xml_head = "<?xml version='1.0' encoding='utf-8'?>\n<linkGrp toDoc='{}' fromDoc='{}'>".format(toDoc, fromDoc)
    xml_tail = "\n</linkGrp>\n"
    xml_body = []
    for bead in (links):
        src_type, src_id = get_link_type_and_id(bead[0])
        tgt_type, tgt_id = get_link_type_and_id(bead[1])
        link = "<link type='{}-{}' xtargets='{};{}' status='auto'/>".format(tgt_type, src_type, tgt_id, src_id)
        xml_body.append(link)
    
    xml_body = "\n".join(xml_body)
    fp = os.path.join(dir, linkDoc)
    with open(fp, 'wt', encoding="utf-8") as f:
        f.write(xml_head + xml_body + xml_tail)
    
def get_link_type_and_id(bead):
    type = len(bead);
    id = ''
    if type > 0:
        id = ' '.join(["1:{}".format(x+1) for x in bead])
    
    return type, id

def write_sent_xml(lines, dir, doc):
    xml_head = "<?xml version='1.0' encoding='utf-8'?>\n<text>\n  <p id=\"1\">\n"
    xml_tail = "\n  </p>\n</text>\n"
    xml_body = []
    for (id, line) in enumerate(lines):
        line = re.sub('&', 'and', line)
        line = re.sub('<|>', '\'', line)
        line = "    <s id=\"1:{}\">{}</s>".format(id+1, line)
        xml_body.append(line)
    
    xml_body = "\n".join(xml_body)
    fp = os.path.join(dir, doc)
    with open(fp, 'wt', encoding="utf-8") as f:
        f.write(xml_head + xml_body + xml_tail)

def write_tmx(src_lines, src_lang, tgt_lines, tgt_lang, links, tmxDoc):
    tmx_head = """<?xml version="1.0" encoding="UTF-8" ?>
<tmx version="1.4">
<header creationtool="BertAlign" creationtoolversion="1.0" segtype="sentence" o-tmf="unknown" adminlang="en-US" srclang="{}" datatype="plaintext" />
<body>""".format(LANG.TMX[src_lang])
    tmx_tail = """</body>
</tmx>"""
    tmx_body = []
    for beads in links:
        src_line = get_line(beads[0], src_lines)
        src_line = convert_line(src_line)
        tgt_line = get_line(beads[1], tgt_lines)
        tgt_line = convert_line(tgt_line)
        tu = """<tu>
<tuv xml:lang="{}"><seg>{}</seg></tuv>
<tuv xml:lang="{}"><seg>{}</seg></tuv>
</tu>""".format(LANG.TMX[src_lang], src_line, LANG.TMX[tgt_lang], tgt_line)
        tmx_body.append(tu)
    
    tmx_body = '\n'.join(tmx_body)
    with open(tmxDoc, 'wt', encoding="utf-8") as f:
        f.write(tmx_head + "\n" + tmx_body + "\n" + tmx_tail)

def convert_line(line):
    line = re.sub(r"&","&amp;",line)
    line = re.sub(r"<","&lt;",line)
    line = re.sub(r">","&gt;",line)
    return line

def get_line(bead, lines):
    line = ''
    if len(bead) > 0:
        line = ' '.join(lines[bead[0]:bead[-1]+1])
    return line
    
def read_lines(path):
    lines = []
    with open(path, 'rt', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lines.append(line)
            
    return lines

def read_alignments(path):
    alignments = []
    with open(path, 'rt', encoding="utf-8") as infile:
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

# Map ISO 639-1 to TMX language code.  
class LANG(object):
    TMX = {
        'zh': 'zh-CN',
        'en': 'en-US',
        'ar': 'ar-UAE',
        'de': 'de-DE',
        'fr': 'fr-FR',
        'nl': 'nl-NL',
        'it': 'it-IT',
        'ja': 'ja-JP',
        'ru': 'ru-RU',
        'pl': 'pl-PL',
        'es': 'es-ES',
    }
    
def make_dir(converted_alignment_path):
  """
  Make an empty diretory for saving converted alignments. 
  """
  if os.path.isdir(converted_alignment_path):
    shutil.rmtree(converted_alignment_path)
  os.makedirs(converted_alignment_path, exist_ok=True)

if __name__ == '__main__':
    main()
