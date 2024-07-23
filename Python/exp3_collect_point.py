#!/usr/bin/env python3
# copy dataset for randomly selected synths to new work area for convenience
import shutil
import os

base_path = "path/to/original/synth/datasets"

# modify for each point
n_point = 32
dest_path = "exp3_points"
data_file =  f"ex3_{n_point}_synths.csv"
point_name = f"exp3_{n_point}"

synths = [s[:-1] for s in open(data_file)]

point_path = dest_path+"/"+point_name
if not os.path.exists(point_path):
    os.mkdir(point_path)
for s in synths:
    shutil.copytree(base_path+s, point_path+"/"+s)
