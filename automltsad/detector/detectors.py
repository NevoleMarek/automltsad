import logging

import numpy as np

from automltsad.detector import BaseDetector

_LOGGER = logging.getLogger(__name__)


def check_if_fitted(model):
    if not model._fitted:
        raise Exception(f'{model.__class__.__name__} is not fitted.')


class TrivialDetector(BaseDetector):
    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: np.ndarray):
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        self._fitted = True
        self._mean = np.mean(X)
        self._std = np.std(X)
        self._threshold = np.quantile((X - self._mean) / self._std, 0.99)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        scores = self.predict_anomaly_scores(X)
        return np.abs(scores) > self._threshold

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        check_if_fitted(self)

        return (X - self._mean) / self._std
