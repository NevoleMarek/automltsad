import json
import os

import numpy as np
import pandas as pd
import torch
import yaml
from config import DATA_DIR, DATASET_DIR, MODEL_DIR

from automltsad.detectors import VAE, AutoEncoder
from automltsad.transform import MinMaxScaler
from automltsad.utils import (
    reduce_window_scores,
    sliding_window_sequences,
    to_time_series_dataset,
)


def get_yaml_config(path):
    with open(path + '.yaml') as f:
        return yaml.full_load(f)


def read_file(fn):
    with open(fn) as f:
        return f.readlines()


def get_dataset_seasonality():
    lines = read_file(DATA_DIR + 'numenta_period.txt')
    dct = {t[0]: int(t[1]) for t in [l.strip('\n').split(', ') for l in lines]}
    return dct


def prepare_data(filename, scale, window, size):
    with open(DATA_DIR + 'datasets/numenta/combined_labels.json', 'r') as f:
        labels = json.load(f)

    with open(DATA_DIR + 'datasets/numenta/combined_windows.json', 'r') as f:
        window_labels = json.load(f)

    labels = {k.split('/')[1]: v for k, v in labels.items()}
    window_labels = {k.split('/')[1]: v for k, v in window_labels.items()}

    ts = pd.read_csv(DATA_DIR + 'datasets/numenta/' + filename)
    ts['label'] = 0

    train = ts.loc[: len(ts) // 2]
    test = ts.loc[len(ts) // 2 :]

    for tstamp, (s, e) in zip(labels[filename], window_labels[filename]):
        if tstamp in test['timestamp'].values:
            test.loc[
                (test['timestamp'] >= s) & (test['timestamp'] <= e), 'label'
            ] = 1
    label = test['label'].values
    ts = pd.read_csv(DATASET_DIR + 'numenta/' + filename)['value'].values
    ts = to_time_series_dataset(ts)

    if scale:
        slr = MinMaxScaler()
        ts = slr.fit_transform(ts)
    train = ts[:, : len(ts[0]) // 2]
    test = ts[:, len(ts[0]) // 2 :]
    label = np.squeeze(label)

    if window:
        train, test = window_data(train, test, size)

    return train, test, label


def save_result(result, f):
    detector, dataset, metrics = result
    dir_path = MODEL_DIR + f'{detector}/{f}/{dataset}/'
    os.makedirs(dir_path, mode=0o777, exist_ok=True)
    with open(
        dir_path + 'result.yaml',
        'w',
    ) as file:
        yaml.dump(metrics, file)


def window_data(train, test, size):
    train_w = sliding_window_sequences(train, size)
    test_w = sliding_window_sequences(test, size)
    return train_w, test_w


def get_latent_dataset(train, test, dataset):
    model = AutoEncoder.load_from_checkpoint(
        DATA_DIR + 'numenta_ae/' + dataset + '-.ckpt'
    )
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    return (
        model(torch.from_numpy(train.copy()).to(torch.float32))
        .detach()
        .unsqueeze(2)
        .numpy(),
        model(torch.from_numpy(test.copy()).to(torch.float32))
        .detach()
        .unsqueeze(2)
        .numpy(),
    )


def get_latent_dataset_vae(train, test, dataset):
    model = VAE.load_from_checkpoint(
        DATA_DIR + 'numenta_vae/' + dataset + '-.ckpt'
    )
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    return (
        model.encode(torch.from_numpy(train.copy()).to(torch.float32))
        .detach()
        .unsqueeze(2)
        .numpy(),
        model.encode(torch.from_numpy(test.copy()).to(torch.float32))
        .detach()
        .unsqueeze(2)
        .numpy(),
    )


def get_autoencoder(dataset):
    return AutoEncoder.load_from_checkpoint(
        DATA_DIR + 'numenta_ae/' + dataset + '-.ckpt'
    )


def get_vae(dataset):
    return VAE.load_from_checkpoint(
        DATA_DIR + 'numenta_vae/' + dataset + '-.ckpt'
    )


def skip_task(task, experiment, file):
    detector, dataset, _ = task
    return os.path.exists(
        f'{MODEL_DIR}{detector}/numenta_{experiment}/{dataset}/{file}'
    )
