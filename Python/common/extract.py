import os
import glob
import librosa as lr
import numpy as np


def extract_folder(f):
    print("extracting "+f)
    wav_path = f+"/wav"
    mfcc_path = f+"/mfcc/"
    if not os.path.exists(mfcc_path): os.mkdir(mfcc_path)
    files = glob.glob(wav_path+"/*.wav")
    for f in files:
        dest_fname = mfcc_path+"/"+os.path.splitext(os.path.basename(f))[0]+".mfcc"
        (y, sr) = lr.load(f, sr=None)
        mfcc = lr.feature.mfcc(y=y, sr=sr)
        np.save(dest_fname, mfcc)
