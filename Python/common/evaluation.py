import librosa as lr
import numpy as np
import torch
import torch.nn as nn
import os, subprocess, uuid
from .extract import extract_folder
from .config import device, sclang_path, mono_render_script, ensemble_render_script
from .models import load_model

def mfccd(a, b):
    return np.mean(np.sqrt(np.mean(np.square(a - b), -1)))
    
def get_param_dist(y, predicted):
    mse = nn.MSELoss()
    l  = mse(y, predicted)
    return l.item()

def get_mfcc_dists(mfcc, predicted, test_path, synth_id = None):
    num_examples = mfcc.shape[0]
    test_dir = test_path + "/" + str(uuid.uuid1())
    os.mkdir(test_dir)
    predicted = predicted.detach().cpu().numpy()
    labels_file_path = f"{test_dir}/params.csv"
    np.savetxt(labels_file_path, predicted[:num_examples], delimiter = ",")
    if synth_id is None:
        subprocess.run([sclang_path, 
                       mono_render_script,
                       labels_file_path, 
                       f"{test_dir}/wav/"])
    else:
        subprocess.run([sclang_path, 
                   ensemble_render_script,
                   synth_id, labels_file_path, 
                   f"{test_dir}/wav/"])
    extract_folder(f"{test_dir}")
    errors = []
    for i in range(num_examples):
        predicted_mfcc = np.load(f"{test_dir}/mfcc/{i}.mfcc.npy")
        original_mfcc = mfcc[i].cpu().numpy()
        mfcc_dist = mfccd(predicted_mfcc, original_mfcc)
        errors.append(mfcc_dist)
    return errors
    
# in-domain (exp1)
def test_model(model, test_data, test_path, synth_id):
    x,y = next(iter(test_data))
    x = x.to(device)
    y = y.to(device)
    predicted = model(x)
    param_dist = get_param_dist(y, predicted)
    mfcc_dists = get_mfcc_dists(x, predicted, test_path, synth_id)
    return (param_dist, np.mean(mfcc_dists))

# out-of-domain (exp3)
def eval_model(model, test_data, model_path, synth_id = None):
    test_dir = "/tmp/"+str(uuid.uuid1())
    num_examples = test_data.shape[0]
    mfcc = torch.from_numpy(test_data).to(device)
    pred = model(mfcc)
    os.mkdir(test_dir)
    predicted = pred.detach().cpu().numpy()
    labels_file_path = f"{test_dir}/params.csv"
    np.savetxt(labels_file_path, predicted[:num_examples], delimiter=",")
    subprocess.run([sclang_path, 
                       ensemble_render_script, 
                       synth_id,
                       labels_file_path, 
                       f"{test_dir}/wav/"])
    extract_folder(f"{test_dir}")
    errors = []
    for i in range(num_examples):
        predicted_mfcc = np.load(f"{test_dir}/mfcc/{i}.mfcc.npy")
        original_mfcc = test_data[i]
        mfcc_dist = mfccd(predicted_mfcc,original_mfcc)
        errors.append(mfcc_dist)
    return errors

def get_accuracy(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data, labels = data
            outputs = model(data.to(device))
            _, predicted = torch.max(outputs, 1)
            _, ground_truth = torch.max(labels, 1)
            ground_truth = ground_truth.to(device)
            total += labels.size(0)
            correct += (predicted == ground_truth).sum().item()
    return correct / total



def eval_ensemble_best(test_data, ensemble_path, synth_ids, model_name, name):
    num_examples = test_data.shape[0]
    ensemble_errs = []
    for i, synth_id in enumerate(synth_ids):
        synth_path = ensemble_path+synth_id+"/"
        model = load_model(synth_path, model_name)
        mfcc = torch.from_numpy(test_data).to(device)
        predicted = model(mfcc)
        errs = get_mfcc_dists(mfcc, predicted, synth_path, synth_id)
        ensemble_errs.append(errs)
    best_err = []
    best_model = []
    for i in range(num_examples):
        results = [x[i] for x in ensemble_errs]
        results = np.array(results)
        amin = np.argmin(results)
        print(i, synth_ids[amin])
        best_err.append(np.min(results))
        best_model.append(amin)
    np.savetxt(f"{name}_results.csv", best_err, delimiter=",")
    return best_err, best_model