import csv
import logging
import warnings
from time import perf_counter

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from config import DATA_DIR, DATASET_DIR, TEST_DATASETS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from automltsad.detectors import AutoEncoder
from automltsad.detectors.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    ModelCheckpoint,
)
from automltsad.transform import MinMaxScaler
from automltsad.utils import sliding_window_sequences, to_time_series_dataset

warnings.filterwarnings('ignore')
_LOGGER = logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


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


def main():
    # Load configs and metadata
    datasets = [d.strip('\n') for d in read_file(TEST_DATASETS)]
    datasets_seasonality = get_dataset_seasonality()

    res = []
    for dataset in tqdm.tqdm(datasets):
        start = perf_counter()
        window_sz = datasets_seasonality[dataset][0]
        train, _, _ = prepare_data(dataset, True, True, window_sz)

        n_s, n_t, n_f = train.shape
        X_tensor = torch.from_numpy(train.copy()).view(n_s, n_t * n_f)
        X_train, X_valid = train_test_split(
            X_tensor, test_size=0.3, shuffle=False
        )
        train_loader = DataLoader(
            X_train.to(torch.float32),
            batch_size=256,
            drop_last=True,
        )
        val_loader = DataLoader(
            X_valid.to(torch.float32), batch_size=X_valid.shape[0]
        )

        hidden = [2**i for i in range(3, int(np.log2(200)) + 1)]

        model = AutoEncoder(
            window_sz,
            encoder_hidden=hidden[::-1],
            decoder_hidden=hidden,
            latent_dim=5,
            learning_rate=1e-3,
        )

        trainer_config = dict(
            accelerator='gpu',
            devices=1,
            precision='16',
            max_epochs=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                EarlyStopping('val_loss', patience=1),
                ModelCheckpoint(
                    dirpath=DATA_DIR + 'ae',
                    filename=dataset + '-',
                    monitor='val_loss',
                    save_top_k=1,
                    mode='min',
                ),
            ],
        )

        trainer = pl.Trainer(**trainer_config)
        trainer.fit(model, train_loader, val_loader)
        end = perf_counter()
        elapsed = end - start
        res.append([dataset, elapsed])

    with open(DATA_DIR + 'ae/elapsed_time.csv', mode='a') as csv_file:
        fieldnames = ['dataset', 'time']
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(fieldnames)
        writer.writerows(res)


if __name__ == '__main__':
    main()
