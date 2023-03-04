import csv
import os
import warnings
from time import perf_counter

import numpy as np
import tqdm
from config import (
    CONFIG_DIR,
    MODEL_DIR,
    NAME_TO_MODEL,
    RESULTS_DIR,
    TEST_DATASETS,
)

from automltsad.metrics import (
    auc,
    f1_pa,
    f1_pa_auc_score,
    precision_recall_curve,
    roc_auc_score,
)from utils import (
    get_autoencoder,
    get_dataset_seasonality,
    get_latent_dataset,
    get_yaml_config,
    prepare_data,
    read_file,
    save_result,
)
from automltsad.utils import reduce_window_scores

warnings.filterwarnings('ignore')

EXPERIMENT = 'baseline'


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
        # writer.writerow(fieldnames)
        writer.writerows(res)


if __name__ == '__main__':
    main()
