# Experiments

The directory contains all scripts and configurations used for conducting experiments. All outputs and results are also present.

## Contents

Jupyter notebooks containing explorations of datasets/results of experiments:

- [Dataset exploration](./00_dataset_exploration.ipynb)

- [Window size selection](./01_window_size_importance.ipynb)

- [Time series specific meta features extraction](./03_tsfresh_features.ipynb)

- [UCR results exploration](./02_results_exploration.ipynb)

- [NAB results exploration](./04_numenta_results_exploration.ipynb)


Benchmark scripts:

- Script for training AutoEncoder for each dataset [UCR](./main_autoencoding.py)/[NAB](./numenta_benchmarks/main_autoencoding.py)

- Script for training VAE for each dataset [UCR](./main_vae.py)/ [NAB](./numenta_benchmarks/main_vae.py)

- Baseline experiment [UCR](./main_baseline.py)/[NAB](./numenta_benchmarks/main_baseline.py)

- US experiment [UCR](./main_unsupervised.py)/[NAB](./numenta_benchmarks/main_unsupervised.py)

- EUS experiment [UCR](./main_unsupervised_ae.py)/[NAB](./numenta_benchmarks/main_unsupervised_ae.py)

- EUSv experiment [UCR](./main_unsupervised_vae.py)/[NAB](./numenta_benchmarks/main_unsupervised_vae.py)

- EUS-AE experiment [UCR](./main_unsupervised_aev2.py)/[NAB](./numenta_benchmarks/main_unsupervised_aev2.py)

- EUS-VAE experiment [UCR](./main_unsupervised_vaev2.py)/[NAB](./numenta_benchmarks/main_unsupervised_vaev2.py)

- META experiment [UCR](./main_meta.py)/[NAB](./numenta_benchmarks/main_meta.py)

## How to run

If desirable to rerun the experiments, the datasets need to be downloaded (400 MB) from [here](https://drive.google.com/drive/folders/1OO1GgRWAo-lzXnQixpRVrmrirA-Gu_uM?usp=share_link). Place  datasets folder in data directory.

With Poetry environment setup you can run any script using: ```poetry run python script.py```

Path for the environment is in most cases as follows: ```/root/.cache/pypoetry/virtualenvs/...```

Additionally, path can be obtained running  ```poetry env info``` command.

All Jupyter notebooks were written using VSCode. VSCode requires a path to python executable in the Poetry env to recognise the kernel. It should be in ```/root/.cache/pypoetry/virtualenvs/FILL_YOUR_ENV_HERE/bin/python``` .

## Disclaimer

The experiment scripts are made for both CPU and GPU experiments. Models being tested in each experiment are given in configuration files [here](/../../tree/main/experiments/config). Nearly all scripts use multiprocessing to speed up computation. To control number of processes a variable `NUM_WORKERS` is used. Running the same number of workers for GPU experiments as for CPU is very likely to end in Out Of Memory error.

Experiments take several hundered hours of CPU and GPU time.