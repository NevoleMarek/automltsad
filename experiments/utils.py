import os

import numpy as np
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
        DATA_DIR + 'ae/' + dataset + '-.ckpt'
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
    model = VAE.load_from_checkpoint(DATA_DIR + 'vae/' + dataset + '-.ckpt')
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
        DATA_DIR + 'ae/' + dataset + '-.ckpt'
    )


def get_vae(dataset):
    return VAE.load_from_checkpoint(DATA_DIR + 'vae/' + dataset + '-.ckpt')


def skip_task(task, experiment, file):
    detector, dataset, _ = task
    return os.path.exists(
        f'{MODEL_DIR}{detector}/{experiment}/{dataset}/{file}'
    )
