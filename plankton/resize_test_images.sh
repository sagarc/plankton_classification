import os
import sys
import subprocess

if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

cmd = "convert -resize 256x256\! "

os.chdir(fo)
imgs = os.listdir(fi)
for img in imgs:
    md = ""
    md += cmd
    md += fi + "/" + img
    md += " " + fo + "/" + img
    os.system(md)
