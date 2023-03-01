import csv
import datetime
import multiprocessing
import os
from time import perf_counter

import numpy as np
import optuna
import tqdm
import yaml
from config import (
    CONFIG_DIR,
    DATA_DIR,
    DATASET_DIR,
    MODEL_DIR,
    NAME_TO_MODEL,
    TEST_DATASETS,
    get_hparams,
)
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from automltsad.metrics import (
    em_feature_subsampling_auc_score,
    excess_mass_auc_score,
    mass_volume_auc_score,
    mv_feature_subsampling_auc_score,
)
from automltsad.transform import MinMaxScaler
from automltsad.utils import sliding_window_sequences, to_time_series_dataset

EXPERIMENT = 'unsupervised'


def get_yaml_config(path):
    with open(path + '.yaml') as f:
        return yaml.full_load(f)


def read_file(fn):
    with open(fn) as f:
        return f.readlines()


def get_dataset_seasonality():
    lines = read_file(DATA_DIR + 'period.txt')
    dct = {
        t[0]: (t[1], t[2]) for t in [l.strip('\n').split(', ') for l in lines]
    }
    for k, v in dct.items():
        if v[0] != 'None' and v[1] != 'None':
            dct[k] = (int(v[0]), int(v[1]))
        else:
            dct[k] = (0, 0)
    return dct


def prepare_data(filename, scale, window, size):
    test_start, anomaly_start, anomaly_end = [
        int(i) for i in filename.split('.')[0].split('_')[-3:]
    ]
    filename = filename.strip('\n')
    ts = np.loadtxt(f'{DATASET_DIR}/{filename}')
    ts = to_time_series_dataset(ts)
    if scale:
        slr = MinMaxScaler()
        ts = slr.fit_transform(ts)
    train = ts[:, :test_start]
    test = ts[:, test_start:]
    label = np.zeros_like(test)
    label[:, anomaly_start - test_start : anomaly_end - test_start] = 1
    label = np.squeeze(label)
    if window:
        train, test = window_data(train, test, size)
    return train, test, label


def window_data(train, test, size):
    train_w = sliding_window_sequences(train, size)
    test_w = sliding_window_sequences(test, size)
    return train_w, test_w


def objective(trial, detector, dataset, det_cfg, window_sz, metric):
    # Prepare hyperparams for model
    hps = get_hparams(trial, detector, det_cfg)
    # Get model
    model = NAME_TO_MODEL[detector]

    # Prepare data
    train, test, labels = prepare_data(
        dataset, scale=True, window=det_cfg['window'], size=window_sz
    )

    # Train model
    det = model(**hps)
    match metric:
        case 'em':
            return em_feature_subsampling_auc_score(
                det, train, test, window_sz, n_tries=30, mc_samples_count=65536
            )
        case 'mv':
            return mv_feature_subsampling_auc_score(
                det,
                train,
                test,
                window_sz,
                n_tries=30,
                alphas_count=128,
                mc_samples_count=65536,
            )
        case _:
            raise ValueError('Wrong metric for unsupervised optimization.')


def process_task(task):
    detector, dataset, window_sz = task

    # Get config
    det_cfg = get_yaml_config(os.path.join(MODEL_DIR, detector, 'config'))

    if not window_sz:
        window_sz = 16

    metrics = {'mv': {}, 'em': {}}
    for metric in ['em', 'mv']:

        start = perf_counter()
        # Workaround to be able to pass multiple arguments to objective
        func = lambda trial: objective(
            trial, detector, dataset, det_cfg, window_sz, metric
        )

        # Optimize
        study = optuna.create_study(
            direction='maximize' if metric == 'em' else 'minimize',
            study_name=f'Unsupervised-{metric}-{detector}',
            sampler=TPESampler(),
        )
        study.optimize(func, n_trials=50, timeout=1800)
        end = perf_counter()
        metrics[metric]['hps'] = study.best_params
        metrics[metric]['value'] = study.best_value
        metrics[metric]['time'] = end - start
    return detector, dataset, metrics


def save_result(result):
    detector, dataset, metrics = result
    dir_path = MODEL_DIR + f'{detector}/unsupervised/{dataset}/'
    os.makedirs(dir_path, mode=0o777, exist_ok=True)
    with open(
        dir_path + 'result.yaml',
        'w',
    ) as file:
        yaml.dump(metrics, file)


def main():
    MAX_WORKERS = 40
    # Load configs and metadata
    automl_cfg = get_yaml_config(CONFIG_DIR + EXPERIMENT)
    datasets = [d.strip('\n') for d in read_file(TEST_DATASETS)]
    datasets_seasonality = get_dataset_seasonality()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tasks = []
    for detector in automl_cfg['detectors']:
        for dataset in datasets:
            tasks.append([detector, dataset, datasets_seasonality[dataset][0]])

    with multiprocessing.Pool(MAX_WORKERS) as p:
        for result in tqdm.tqdm(
            p.imap_unordered(process_task, tasks, chunksize=4),
            total=len(tasks),
        ):
            save_result(result)


if __name__ == '__main__':
    main()
