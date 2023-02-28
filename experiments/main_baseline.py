import csv
import os
from time import perf_counter

import numpy as np
import tqdm
import yaml
from config import (
    CONFIG_DIR,
    DATA_DIR,
    DATASET_DIR,
    MODEL_DIR,
    NAME_TO_MODEL,
    RESULTS_DIR,
    TEST_DATASETS,
    TRAIN_DATASETS,
)

from automltsad.detectors import KNN, LOF, IsolationForestAD
from automltsad.metrics import (
    auc,
    f1_pa,
    f1_pa_auc_score,
    precision_recall_curve,
    roc_auc_score,
)
from automltsad.transform import MinMaxScaler
from automltsad.utils import (
    reduce_window_scores,
    sliding_window_sequences,
    to_time_series_dataset,
)

EXPERIMENT = 'baseline'


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


def evaluate_model(scores, labels):
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    idx = np.nanargmax(f1)
    step = max(len(t) // 100, 1)
    ts = t[::step]
    f1_pas = [f1_pa(labels, scores, th) for th in ts]
    f1pat = f1_pa(labels, scores, t[idx])
    idx_pa = np.nanargmax(f1_pas)
    f1pa = f1_pas[idx_pa]
    aucpr = auc(r, p)
    f1_pa_t_auc = f1_pa_auc_score(labels, scores, t[idx])
    f1_pa_ts_auc = f1_pa_auc_score(labels, scores, ts[idx_pa])
    roc_auc = roc_auc_score(labels, scores)
    return f1[idx], f1_pa_t_auc, f1pa, f1_pa_ts_auc, aucpr, roc_auc


def main():
    automl_cfg = get_yaml_config(CONFIG_DIR + EXPERIMENT)
    datasets = [d.strip('\n') for d in read_file(TEST_DATASETS)]
    datasets_seasonality = get_dataset_seasonality()

    res = []

    for detector in tqdm.tqdm(
        automl_cfg['detectors'], desc='Detector', leave=False
    ):
        # Get model and config
        det_cfg = get_yaml_config(os.path.join(MODEL_DIR, detector, 'config'))
        model = NAME_TO_MODEL[detector]

        for dataset in tqdm.tqdm(datasets, desc='Dataset', leave=False):
            start = perf_counter()
            # Window size slightly larger than 1 period
            window_sz = int(datasets_seasonality[dataset][0])
            if not window_sz:
                window_sz = 16
            # Prepare data
            train, test, labels = prepare_data(
                dataset, scale=True, window=det_cfg['window'], size=window_sz
            )

            # Train model
            det = model()
            det.fit(train)

            # Predict scores
            scores = det.predict_anomaly_scores(test)

            # Reduce scores if using windows
            if det_cfg['window']:
                scores = reduce_window_scores(scores, window_sz)

            # Evaluate performance on test
            r = evaluate_model(scores, labels)
            end = perf_counter()
            res.append([detector, dataset] + [*r] + [end - start])

    with open(RESULTS_DIR + f'{EXPERIMENT}_results.csv', mode='a') as csv_file:
        fieldnames = [
            'detector',
            'dataset',
            'f1',
            'f1_pa_auc',
            'f1_pa',
            'f1_pa_ts_auc',
            'aucpr',
            'aucroc',
            'time',
        ]
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(fieldnames)
        writer.writerows(res)


if __name__ == '__main__':
    main()
