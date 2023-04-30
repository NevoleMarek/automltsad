import csv
import logging
import warnings
from time import perf_counter

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from config import DATA_DIR, TEST_DATASETS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import get_dataset_seasonality, prepare_data, read_file

from automltsad.detectors import VAE
from automltsad.detectors.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    ModelCheckpoint,
)

warnings.filterwarnings('ignore')
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


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
        val_loader = DataLoader(X_valid.to(torch.float32), batch_size=256)

        l = int(np.log2(window_sz) - 0.0001)
        hidden = [8, 2 ** ((l + 3) // 2), 2**l]

        model = VAE(
            window_sz,
            encoder_hidden=hidden[::-1],
            decoder_hidden=hidden,
            latent_dim=5,
            learning_rate=0.001,
        )

        trainer_config = dict(
            accelerator='gpu',
            devices=[1],
            precision='16',
            max_epochs=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            callbacks=[
                ModelCheckpoint(
                    dirpath=DATA_DIR + 'vae',
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

    with open(DATA_DIR + 'vae/elapsed_time.csv', mode='w') as csv_file:
        fieldnames = ['dataset', 'time']
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')
        writer.writerow(fieldnames)
        writer.writerows(res)


if __name__ == '__main__':
    main()
