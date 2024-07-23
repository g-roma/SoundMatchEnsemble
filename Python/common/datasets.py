import numpy as np
import glob
from torch.utils.data import Dataset


class MFCCDataset(Dataset):
    def __init__(self, synth_path):
        self.mfcc_path =  synth_path + "/mfcc"
        self.labels_path = synth_path + "/labels"
        self.transform = None

    def __len__(self):
        files = glob.glob(self.mfcc_path + "/*.mfcc.npy")
        return len(files)

    def __getitem__(self, idx):
        label_file = self.labels_path + "/" + str(idx) + ".csv"
        labels = np.genfromtxt(label_file, 
                    delimiter = ",", 
                    dtype = np.float32, usemask = False)

        labels = labels[:-1]
        mfcc_file = self.mfcc_path + "/" + str(idx) + ".mfcc.npy"
        mfcc = np.load(mfcc_file, allow_pickle = False).astype(np.float32)
        return mfcc, labels
   
class PredictSameDataset(Dataset):
    def __init__(self, base_path, synth_ids, fr, to):
        self.paths = []
        self.labels = []
        for i, s in enumerate(synth_ids):
            label= [0.0 for i in synth_ids]
            label[i] = 1.0
            paths = [f"{base_path}/{s}/mfcc/{j}.mfcc.npy" 
                for j in range(fr, to)]
            labels = np.array([label.copy() for x in paths])
            self.paths.extend(paths)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mfcc = np.load(self.paths[idx] , allow_pickle = False).astype(np.float32)
        return mfcc, self.labels[idx].astype(np.float32)


class PredictBestDataset(Dataset):
    def __init__(self, base_path, filename, synth_ids):
        data = [x.split(",") for x in open(filename)]
        self.paths = []
        self.labels = []
        for d in data:
            self.paths.append(f"{base_path}/mfcc/{d[0]}") 
            label= [0.0 for i in synth_ids]
            label[int(d[1][:-1])] = 1.0
            self.labels.append(np.array(label))
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mfcc = np.load(self.paths[idx] , allow_pickle=False).astype(np.float32)
        return mfcc, self.labels[idx].astype(np.float32)

