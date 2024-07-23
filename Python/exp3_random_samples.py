#!/usr/bin/env python3
# sample "points" (random ensembles) for evaluation at different ensemble sizes
import random

synths = []

for s1 in [0,1]:
    for s2 in [0,1]:
        for m1 in [0,1,2]:
            for m2 in [0,1,2]:
                for m3 in [0,1,2]:
                    for f1 in [0,1,2]:
                        synthid = f"{s1}{s2}{m1}{m2}{m3}1{f1}"
                        synths.append(synthid)

def save_list(data, name):
    with open(name,'w') as out_file:
        for n in data:
            out_file.write(n+"\n")

save_list(random.sample(synths,8), "ex3_8_synths.csv")
save_list(random.sample(synths,16), "ex3_16_synths.csv")
save_list(random.sample(synths,32), "ex3_32_synths.csv")
save_list(random.sample(synths,64), "ex3_64_synths.csv")
save_list(random.sample(synths,128), "ex3_128_synths.csv")
save_list(random.sample(synths,256), "ex3_256_synths.csv")
save_list(synths, "ex3_324_synths.csv")

