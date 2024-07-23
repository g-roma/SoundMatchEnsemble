#!/usr/bin/env python3
# evaluation with 324 synths ensemble
import numpy as np
import os, glob
from common.config import  nsynth_valid
from common.evaluation import eval_ensemble_best
from common.data import get_test_data

ensemble_path = "path/to/ensemble"
synth_ids = [os.path.basename(x) for x in glob.glob(ensemble_path+"*")]
model_name = "cnn_model.pt"
test_data, filenames = get_test_data(nsynth_valid)
result, models = eval_ensemble_best(test_data, ensemble_path, synth_ids, model_name, "324_synths")
np.savetxt("best_models_324.csv", models, delimiter=",")    
