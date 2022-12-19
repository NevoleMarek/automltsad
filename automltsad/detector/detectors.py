import logging

import numpy as np

from automltsad.detector import BaseDetector
from automltsad.validation import (
    check_if_fitted,
    check_one_sample,
    validate_data_format,
)

_LOGGER = logging.getLogger(__name__)


class TrivialDetector(BaseDetector):
    '''
    TrivialDetector is a simple detector implementation.

    TrivialDetector standardizes data based on mean and std of training data.
    Anomaly score is the number of stds from mean value of training data.
    Threshold is determined as 1-contamination-th quantile of standardized
    data.


    Parameters
    ----------
    contamination : float
        Contamination parameter is used to select threshold.
    '''

    def __init__(self, contamination: float) -> None:
        self._fitted = False
        self.contamination = contamination

    def fit(self, X: np.ndarray):
        '''
        Fit the trivial detector on training dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self: TrivialDetector
            The fitted trivial detector.
        '''
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_format(X)
        check_one_sample(X)
        self._fitted = True
        self._mean = np.mean(X, axis=1)
        self._std = np.std(X, axis=1)
        self._threshold = np.quantile(
            (X - self._mean) / self._std, 1 - self.contamination
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, n_timepoints, n_features)
            Labels of data points
        '''
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        validate_data_format(X)
        check_one_sample(X)
        check_if_fitted(self)

        scores = self.predict_anomaly_scores(X)
        return scores > self._threshold

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_timepoints, n_features)
            Anomaly scores.
        '''
        validate_data_format(X)
        check_one_sample(X)
        check_if_fitted(self)

        return np.abs((X - self._mean) / self._std)
