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
    MODEL_DIR,
    NAME_TO_MODEL,
    PYT_MODELS,
    TEST_DATASETS,
    get_hparams,
)
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping
from utils import (
    get_autoencoder,
    get_dataset_seasonality,
    get_latent_dataset,
    get_yaml_config,
    prepare_data,
    read_file,
    save_result,
)

from automltsad.metrics import (
    auc,
    excess_mass_auc_score,
    f1_pa,
    f1_pa_auc_score,
    mass_volume_auc_score,
    precision_recall_curve,
    roc_auc_score,
)
from automltsad.utils import reduce_window_scores

warnings.filterwarnings('ignore', 'Solver terminated early.*')

EXPERIMENT = 'unsupervised_aev2'


def objective(trial, detector, dataset, det_cfg, window_sz, metric):
    # Prepare hyperparams for model
    hps = get_hparams(trial, detector, det_cfg)
    # Get model
    model = NAME_TO_MODEL[detector]

    # Prepare data
    train, test, labels = prepare_data(
        dataset, scale=True, window=det_cfg['window'], size=window_sz
    )
    train_l, test_l = get_latent_dataset(train, test, dataset)
    ae = get_autoencoder(dataset)
    if detector in PYT_MODELS:
        hps['trainer_config'] = dict(
            accelerator='auto',
            devices=1,
            precision='16',
            max_epochs=100,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[
                PyTorchLightningPruningCallback(trial, 'val_loss'),
                EarlyStopping('val_loss', patience=1),
            ],
        )
        hps['window_size'] = window_sz
        hps['seq_len'] = window_sz
        hps['out_len'] = window_sz - 1
        hps['label_len'] = hps['out_len'] // 2

    # Train model
    det = model(**hps)
    det.fit(train)

    # Predict scores
    scores = det.predict_anomaly_scores(test)
    match metric:
        case 'em':
            return excess_mass_auc_score(
                det,
                test_l,
                scores,
                t_count=256,
                mc_samples_count=65536,
                decoder=ae,
            )
        case 'mv':
            return mass_volume_auc_score(
                det,
                test_l,
                scores,
                alphas_count=128,
                mc_samples_count=65536,
                decoder=ae,
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
    for metric in ['mv', 'em']:
        start = perf_counter()
        # Workaround to be able to pass multiple arguments to objective
        func = lambda trial: objective(
            trial, detector, dataset, det_cfg, window_sz, metric
        )

        # Optimize
        if detector in PYT_MODELS:
            opt_params = dict(timeout=300)
        else:
            opt_params = dict(n_trials=30, timeout=1800)

        study = optuna.create_study(
            direction='maximize',
            study_name=f'Unsupervised-aev2-{detector}',
            sampler=TPESampler(),
            pruner=SuccessiveHalvingPruner(),
        )
        study.optimize(func, **opt_params)
        end = perf_counter()
        metrics[metric]['hps'] = study.best_params
        metrics[metric]['value'] = study.best_value
        metrics[metric]['time'] = end - start
    return detector, dataset, metrics


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
        MODEL_DIR + f'{detector}/{EXPERIMENT}/{dataset}/result'
    )[metric]['hps']
    det_cfg = get_yaml_config(os.path.join(MODEL_DIR, detector, 'config'))

    if not window_sz:
        window_sz = 16

    # Get model
    model = NAME_TO_MODEL[detector]

    # Prepare data
    train, test, labels = prepare_data(
        dataset, scale=True, window=det_cfg['window'], size=window_sz
    )
    train, test = get_latent_dataset(train, test, dataset)

    if detector in PYT_MODELS:
        hps['trainer_config'] = dict(
            accelerator='auto',
            devices=1,
            precision='16',
            max_epochs=100,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[
                PyTorchLightningPruningCallback(trial, 'val_loss'),
                EarlyStopping('val_loss', patience=1),
            ],
        )
        hps['window_size'] = window_sz
        hps['seq_len'] = window_sz
        hps['out_len'] = window_sz - 1
        hps['label_len'] = hps['out_len'] // 2

    for metric in ['em', 'mv']:
        # Train model
        det = model(**hps)
        det.fit(train)

        # Predict scores
        scores = det.predict_anomaly_scores(test)

        # Reduce scores if using windows
        if det_cfg['window']:
            scores = reduce_window_scores(scores, window_sz)

        # Evaluate performance on test
        r = evaluate_model(scores, labels)
        dir_path = MODEL_DIR + f'{detector}/{EXPERIMENT}/{dataset}/'
        with open(
            dir_path + f'{metric}.yaml',
            'w',
        ) as file:
            yaml.dump(r, file)


def main():
    MAX_WORKERS = 20
    # Load configs and metadatad
    automl_cfg = get_yaml_config(CONFIG_DIR + EXPERIMENT)
    datasets = [d.strip('\n') for d in read_file(TEST_DATASETS)]
    datasets_seasonality = get_dataset_seasonality()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Parallel optimization
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

    # Load models, best hyperparams and evaluate using supervised metrics
    with multiprocessing.Pool(MAX_WORKERS) as p:
        for result in tqdm.tqdm(
            p.imap_unordered(process_evaluation, tasks),
            total=len(tasks),
        ):
            continue


if __name__ == '__main__':
    main()
