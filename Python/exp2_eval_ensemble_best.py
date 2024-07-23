#!/usr/bin/env python3
# evaluation of "select best" ensemble strategy
import numpy as np
from common.data import get_test_data
from common.config import nsynth_valid, ensemble_synth_ids
from common.evaluation import eval_ensemble_best

ensemble_path = "path/to/ensemble/datasets"
model_name = "exp2_ensemble_model.pt"

test_data, filenames = get_test_data(nsynth_valid)
result, models = eval_ensemble_best(test_data, ensemble_path, 
                                    ensemble_synth_ids,  model_name, "select_best")
print("--------------")
print(np.median(result))

