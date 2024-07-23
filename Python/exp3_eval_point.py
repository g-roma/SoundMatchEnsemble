#!/usr/bin/env python3
# evaluation of a set of randomly selected synths
import numpy as np
import os, glob
from common.config import  nsynth_valid
from common.evaluation import eval_ensemble_best
from common.data import get_test_data

n_point = 32
point_path = f"path/to/exp3_points/exp3_{n_point}/"
synth_ids = [os.path.basename(x) for x in glob.glob(point_path+"*")]
model_name = "cnn_model.pt"

test_data, filenames = get_test_data(nsynth_valid)
result, models = eval_ensemble_best(test_data, point_path, synth_ids, model_name, f"exp3_point_{n_point}_best")

