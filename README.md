## Sound matching using synthesizer ensembles

This repository contains the code to reproduce the method presented in:

> Roma, G.,Sound matching using synthesizer ensembles. Proceedings of the 27th International Conference on Digital Audio Effects (DAFX 2024)


### Requirements:

- Python 3
- Pytorch
- Numpy
- Librosa
- SuperCollider

### Usage:

SuperCollider code: install the classes in Extensions, and use the scripts to generate datasets and render from Python. The script generate_synth_dataset.scd can be used to test within SuperCollider.

Python code: Use the expN scripts to reproduce experiments in the paper and as examples, the code in common can be used as a library to implement the different worlkflows.
