#!/usr/bin/env python3
# create dataset for "predict best" classifier
import os, uuid
import numpy as np
import torch
from common.models import  load_model
from common.evaluation import get_mfcc_dists
from common.data import get_test_data
from common.config import device, ensemble_synth_ids, nsynth_train

ensemble_path = "path/to/ensemble/datasets"
dataset_fname = 'predict_best_dataset.csv'
model_name = "exp2_ensemble_model.pt"


def make_dataset(train_data):
    num_files = train_data.shape[0]
    ensemble_errs = []
    for i, synth_id in enumerate(ensemble_synth_ids):
        print(synth_id)
        test_dir = "/tmp/"+str(uuid.uuid1())
        os.mkdir(test_dir)
        synth_path = ensemble_path + synth_id+"/"
        model = load_model(synth_path, model_name)
        mfcc = torch.from_numpy(train_data).to(device)
        predicted = model(mfcc)
        errs  = get_mfcc_dists(mfcc, predicted, test_dir, synth_id)
        ensemble_errs.append(errs)
    best_err = []
    best_model = []
    for i in range(num_files):
        print(i)
        results = [x[i] for x in ensemble_errs]
        results = np.array(results)
        amin = np.argmin(results)
        best_err.append(np.min(results))
        best_model.append(amin)
    return best_err, best_model

train_data, filenames = get_test_data(nsynth_train)
err, labels = make_dataset(train_data)
with open(dataset_fname,'w') as out_file:
    for i, l in enumerate(labels):
        out_file.write(filenames[i] + "," + str(l) + "\n")

