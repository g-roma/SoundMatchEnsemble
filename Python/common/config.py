device = "mps"
ensemble_synth_ids = ["0002110","0002111","0002112","0102110",
          "0102111","0102112","1002110","1002111",
          "1002112","1102110","1102111","1102112"]
num_epochs = 100
patience = 10
sclang_path = "/Applications/SuperCollider.app/Contents/MacOS/sclang"
ensemble_render_script = "/path/to/render_synth.scd"
mono_render_script = "/path/to/render_monolithic.scd"
# sounds with pitch = 60 were copied using shell command from nsynth dataset
nsynth_train = "path/to/pitch_60_train/" # 4258 samples
nsynth_valid = "path/to/pitch_60_valid/" # 174 samples
