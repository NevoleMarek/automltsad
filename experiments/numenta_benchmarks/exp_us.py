import multiprocessing
import os
import warnings
from time import perf_counter

import numpy as np
import optuna
import tqdm
import yaml
from config import (
    CONFIG_DIR,
    DATA_DIR,
    MODEL_DIR,
    NAME_TO_MODEL,
    TEST_DATASETS,
    get_hparams,
)
from optuna.samplers import TPESampler
from utils import (
    get_dataset_seasonality,
    get_yaml_config,
    prepare_data,
    read_file,
    save_result,
    skip_task,
)

from automltsad.metrics import (
    auc,
    em_feature_subsampling_auc_score,
    f1_pa,
    f1_pa_auc_score,
    mv_feature_subsampling_auc_score,
    precision_recall_curve,
    roc_auc_score,
)
from automltsad.utils import reduce_window_scores

warnings.filterwarnings('ignore', 'Solver terminated early.*')

EXPERIMENT = 'unsupervised'


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
        study.optimize(func, n_trials=30, timeout=1200)
        end = perf_counter()
        metrics[metric]['hps'] = study.best_params
        metrics[metric]['value'] = study.best_value
        metrics[metric]['time'] = end - start

    save_result((detector, dataset, metrics), 'numenta_' + EXPERIMENT)


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
    return dict(
        f1=f1[idx],
        f1_pa_auc=f1_pa_t_auc,
        f1_pa=f1pa,
        f1_pa_ts_auc=f1_pa_ts_auc,
        aucpr=aucpr,
        aucroc=roc_auc,
    )


def process_evaluation(task):
    detector, dataset, window_sz = task
    # Get configs
    hps = get_yaml_config(
        MODEL_DIR + f'{detector}/numenta_{EXPERIMENT}/{dataset}/result'
    )
    det_cfg = get_yaml_config(os.path.join(MODEL_DIR, detector, 'config'))

    if not window_sz:
        window_sz = 16

    # Get model
    model = NAME_TO_MODEL[detector]

    # Prepare data
    train, test, labels = prepare_data(
        dataset, scale=True, window=det_cfg['window'], size=window_sz
    )

    for metric in ['em', 'mv']:
        # Train model
        det = model(**hps[metric]['hps'])
        det.fit(train)

        # Predict scores
        scores = det.predict_anomaly_scores(test)

        # Reduce scores if using windows
        if det_cfg['window']:
            scores = reduce_window_scores(scores, window_sz)

        # Evaluate performance on test
        r = evaluate_model(scores, labels)
        r['metric'] = hps[metric]['value']
        dir_path = MODEL_DIR + f'{detector}/numenta_{EXPERIMENT}/{dataset}/'
        with open(
            dir_path + f'{metric}.yaml',
            'w',
        ) as file:
            yaml.dump(r, file)


def create_tasks(detectors, datasets, windows, file):
    tasks = []
    for detector in detectors:
        for dataset in datasets:
            task = [detector, dataset, windows[dataset]]
            if not skip_task(task, EXPERIMENT, file):
                tasks.append(task)
    return tasks


def main():
    MAX_WORKERS = 40
    # Load configs and metadata
    automl_cfg = get_yaml_config(CONFIG_DIR + EXPERIMENT)
    path = DATA_DIR + 'datasets/numenta'
    datasets = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')
    ]
    datasets_seasonality = get_dataset_seasonality()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tasks = create_tasks(
        automl_cfg['detectors'], datasets, datasets_seasonality, 'result.yaml'
    )
    with multiprocessing.Pool(MAX_WORKERS) as p:
        for result in tqdm.tqdm(
            p.imap(process_task, tasks),
            total=len(tasks),
        ):
            continue

    tasks = create_tasks(
        automl_cfg['detectors'], datasets, datasets_seasonality, 'em.yaml'
    )
    # Load models, best hyperparams and evaluate using supervised metrics
    with multiprocessing.Pool(MAX_WORKERS) as p:
        for result in tqdm.tqdm(
            p.imap(process_evaluation, tasks),
            total=len(tasks),
        ):
            continue


if __name__ == '__main__':
    main()
