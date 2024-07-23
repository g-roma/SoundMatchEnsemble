#!/usr/bin/env python3
# create 324 synths ensemble dataset
import os, subprocess
import shutil
from pathlib import Path
from distutils.dir_util import copy_tree
from common.extract import extract_folder
from common.config import sclang_path

base_path = "path/to/dataset/"
sc_script = "path/to/generate_synth_dataset.scd"
N = 5000

def generate_dataset(synth_id, out_dir):
    tmp_dir = "/tmp/generation"
    subprocess.run([sclang_path, 
                       sc_script,
                       synth_id, tmp_dir, str(N)],
                       timeout=3*60
               )
    extract_folder(tmp_dir)
    
    copy_tree(tmp_dir+"/mfcc", out_dir+"/mfcc")
    copy_tree(tmp_dir+"/labels", out_dir+"/labels")
    shutil.rmtree(tmp_dir+"/wav/")
    shutil.rmtree(tmp_dir+"/labels/")
    Path(out_dir+"/done.txt").touch()
    print("done")


synths = []

for s1 in [0,1]:
    for s2 in [0,1]:
        for m1 in [0,1,2]:
            for m2 in [0,1,2]:
                for m3 in [0,1,2]:
                    for f1 in [0,1,2]:
                        synthid = f"{s1}{s2}{m1}{m2}{m3}1{f1}"
                        synths.append(synthid)
                        
print(len(synths))                        
for s in synths:
    print(s)
    out_dir = base_path+s
    if not os.path.exists(out_dir+"/done.txt"):
        if not os.path.exists(out_dir):os.mkdir(out_dir)
        try:
            generate_dataset(s, out_dir)
            pass
        except:
            print("ERROR",s )
