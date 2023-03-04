import logging
import multiprocessing
import os
import warnings
from time import perf_counter

import numpy as np
import optuna
import tqdm
from config import (
    CONFIG_DIR,
    DATA_DIR,
    DATASET_DIR,
    MODEL_DIR,
    NAME_TO_MODEL,
    PYT_MODELS,
    TEST_DATASETS,
    TRAIN_DATASETS,
    get_hparams,
)
from joblib import dump, load
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler as MMScaler
from utils import (
    get_dataset_seasonality,
    get_yaml_config,
    prepare_data,
    read_file,
    save_result,
)

from automltsad.automl.metaod import MetaODClass, generate_meta_features
from automltsad.automl.metaod.utility import fix_nan
from automltsad.metrics import auc, precision_recall_curve
from automltsad.utils import reduce_window_scores

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

warnings.filterwarnings('ignore', 'Solver terminated early.*')
warnings.filterwarnings('ignore')

EXPERIMENT = 'meta'


def evaluate_model(scores, labels):
    p, r, t = precision_recall_curve(labels, scores)
    aucpr = auc(r, p)
    return aucpr


def objective(trial, detector, dataset, det_cfg, window_sz):
    # Prepare hyperparams for model
    hps = get_hparams(trial, detector, det_cfg)
    # Get model
    model = NAME_TO_MODEL[detector]

    # Prepare data
    train, test, labels = prepare_data(
        dataset, scale=True, window=det_cfg['window'], size=window_sz
    )

    # Train model
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

    det = model(**hps)
    det.fit(train)

    # Predict scores
    scores = det.predict_anomaly_scores(test)

    # Reduce scores if using windows
    if det_cfg['window']:
        scores = reduce_window_scores(scores, window_sz)

    # Evaluate performance on test
    return evaluate_model(scores, labels)


def process_task(task):
    detector, dataset, window_sz = task

    # Get config
    det_cfg = get_yaml_config(os.path.join(MODEL_DIR, detector, 'config'))

    if not window_sz:
        window_sz = 16

    start = perf_counter()
    # Workaround to be able to pass multiple arguments to objective
    func = lambda trial: objective(
        trial, detector, dataset, det_cfg, window_sz
    )

    # Optimize
    if detector in PYT_MODELS:
        opt_params = dict(timeout=300)
    else:
        opt_params = dict(n_trials=30, timeout=1800)

    study = optuna.create_study(
        direction='maximize',
        study_name=f'Supervised-{detector}',
        sampler=TPESampler(),
        pruner=SuccessiveHalvingPruner(),
    )
    study.optimize(func, **opt_params)
    end = perf_counter()
    metrics = dict(
        hps=study.best_params,
        value=study.best_value,
        time=end - start,
    )
    return detector, dataset, metrics


def process_meta(task):
    dataset, w_sz = task
    meta_features = np.zeros([1, 200])
    w_sz = w_sz if w_sz else 16
    train, test, labels = prepare_data(dataset, True, True, w_sz)
    meta_features, meta_vec_names = generate_meta_features(
        np.concatenate((train, test)).squeeze()
    )
    np.save(f'{DATASET_DIR}/metafeatures/{dataset}.npy', meta_features)


def main():
    ###################################
    # Optimize models on train datasets
    ###################################
    MAX_WORKERS = 40
    # Load configs and metadata
    automl_cfg = get_yaml_config(CONFIG_DIR + EXPERIMENT)
    train_datasets = [d.strip('\n') for d in read_file(TRAIN_DATASETS)]
    test_datasets = [d.strip('\n') for d in read_file(TEST_DATASETS)]
    datasets_seasonality = get_dataset_seasonality()
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    tasks = []
    results = {}
    for detector in automl_cfg['detectors']:
        for dataset in train_datasets:
            results[detector] = {}
            results[detector][dataset] = None
            tasks.append([detector, dataset, datasets_seasonality[dataset][0]])

    print('Computing performance matrix')
    with multiprocessing.Pool(MAX_WORKERS) as p:
        for result in tqdm.tqdm(
            p.imap_unordered(process_task, tasks),
            total=len(tasks),
        ):
            save_result(result)
            results[result[0]][result[1]] = result[2]

    ########################################
    # Create performance matrix from results
    ########################################
    n_datasets = len(train_datasets)
    n_models = len(automl_cfg['detectors'])
    performance_matrix = np.zeros((n_datasets, n_models))

    for i, detector in enumerate(automl_cfg['detectors']):
        for j, dataset in enumerate(train_datasets):
            performance_matrix[j, i] = results[detector][dataset]['value']
    np.save(DATA_DIR + 'meta/perf_mat', performance_matrix)
    performance_matrix = np.load(DATA_DIR + 'meta/perf_mat.npy')

    ############################
    # Meta features for datasets
    ############################
    # Load datasets and generate meta features
    print('Computing meta features')
    datasets = train_datasets + test_datasets
    with multiprocessing.Pool(MAX_WORKERS) as p:
        for result in tqdm.tqdm(
            p.imap_unordered(
                process_meta,
                [(d, datasets_seasonality[d][0]) for d in datasets],
            ),
            total=len(datasets),
        ):
            continue

    meta_mat = np.zeros([len(train_datasets), 200])
    for i, ds in enumerate(train_datasets):
        meta_mat[i, :] = np.load(f'{DATASET_DIR}metafeatures/{ds}.npy')

    # use cleaned and transformed meta-features
    meta_scalar = MMScaler()
    meta_mat_transformed = meta_scalar.fit_transform(meta_mat)
    meta_mat_transformed = fix_nan(meta_mat_transformed)

    dump(meta_scalar, f'{DATA_DIR}/meta/' + 'metascalar.joblib')

    n_train = int(len(train_datasets) * 0.75)

    train_set = performance_matrix[:n_train, :].astype('float64')
    valid_set = performance_matrix[n_train:, :].astype('float64')

    train_meta = meta_mat_transformed[:n_train, :].astype('float64')
    valid_meta = meta_mat_transformed[n_train:, :].astype('float64')

    print('Training metaod')

    learning_rate = [1, 0.1, 0.01, 1e-3, 1e-4]
    factors = [5, 10, 20, 40]
    best_clf = None
    best_l = -np.inf
    for l in learning_rate:
        for f in factors:
            clf = MetaODClass(
                train_set,
                valid_performance=valid_set,
                n_factors=f,
                learning='sgd',
            )
            clf.train(
                n_iter=50,
                meta_features=train_meta,
                valid_meta=valid_meta,
                learning_rate=l,
            )

            l = np.nanargmax(clf.valid_loss_)
            if l > best_l:
                best_clf = clf

    dump(best_clf, f'{DATA_DIR}/meta/train_0.joblib')

    # # load PCA scalar
    meta_scalar = load(os.path.join(DATA_DIR, 'meta/metascalar.joblib'))

    # # # generate meta features
    test_meta_mat = np.zeros([len(test_datasets), 200])
    for i, ds in enumerate(test_datasets):
        test_meta_mat[i, :] = np.load(f'{DATASET_DIR}metafeatures/{ds}.npy')

    # replace nan by 0 for now
    # todo: replace by mean is better as fix_nan
    test_meta_mat = meta_scalar.transform(test_meta_mat)
    test_meta_mat = np.nan_to_num(test_meta_mat, nan=0)

    clf = load(f'{DATA_DIR}/meta/train_0.joblib')
    predict_scores = clf.predict(test_meta_mat)
    best_models = np.nanargmax(predict_scores, axis=1)
    print([automl_cfg['detectors'][m] for m in best_models])


if __name__ == '__main__':
    main()
