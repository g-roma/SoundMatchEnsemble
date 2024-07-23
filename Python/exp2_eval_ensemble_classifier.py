#!/usr/bin/env python3
# evaluation of classifier-based ensemble strategies
import torch
import librosa as lr
import numpy as np
import subprocess
import os, uuid
from common.data import get_test_data
from common.models import CNNModel, load_model
from common.config import device, sclang_path, ensemble_render_script, nsynth_valid, ensemble_synth_ids
from common.evaluation import mfccd

ensemble_path = "path/to/ensemble/datasets"
model_name = "exp2_ensemble_model.pt"

# modify to test predict_best or predict_similar
classifier_model_fname = "exp2_predict_best.pt"
results_fname = "predict_best"
#classifier_model_fname = "exp2_predict_same.pt"
#results_fname = "exp2_predict_same"

def eval_ensemble_classifier(test_data, name):
    test_dir = "/tmp/"+str(uuid.uuid1())
    os.mkdir(test_dir)
    os.mkdir(test_dir+"/mfcc")
    num_examples = test_data.shape[0]
    ensemble_models = [load_model(ensemble_path+"/"+i+"/", model_name) for i in ensemble_synth_ids]
    classifier_model = CNNModel(12).to(device)
    classifier_model.load_state_dict(
    torch.load(classifier_model_fname))    
    classifier_model.eval()

    mfcc = torch.from_numpy(test_data).to(device)
    classifier_out = classifier_model(mfcc)
    _, class_id = torch.max(classifier_out, 1)

    errors = []
    for example in range(num_examples):
        synth_idx = class_id[example]
        synth_id = ensemble_synth_ids[synth_idx]
        synth_model = ensemble_models[synth_idx]
        data = mfcc[example][None,:,:]
        predicted = synth_model(data)
        predicted = predicted.detach().cpu().numpy()
        labels_file_path = f"{test_dir}/params.csv"
        np.savetxt(labels_file_path, predicted[:num_examples], delimiter = ",")
        subprocess.run([sclang_path, 
                           ensemble_render_script, 
                           synth_id,
                           labels_file_path, 
                           f"{test_dir}/wav/"])
        wav_file = f"{test_dir}/wav/0.wav"
        (y, sr) = lr.load(wav_file, sr=None)
        predicted_mfcc = lr.feature.mfcc(y=y, sr=sr)
        original_mfcc = test_data[example]
        mfcc_dist = mfccd(predicted_mfcc,original_mfcc)
        errors.append(mfcc_dist)
        print("---", example)
    np.savetxt(f"{name}_results.csv", errors, delimiter=",")    
    return errors

test_data, filenames = get_test_data(nsynth_valid)
result = eval_ensemble_classifier(test_data, results_fname)
print("--------------")
print(np.median(result))