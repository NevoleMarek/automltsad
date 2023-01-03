import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from automltsad.detector import BaseDetector
from automltsad.transform import MeanVarianceScaler
from automltsad.utils import reduce_window_scores, sliding_window_sequences
from automltsad.validation import (
    check_if_fitted,
    check_one_sample,
    validate_data_2d,
    validate_data_3d,
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
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self: TrivialDetector
            The fitted trivial detector.
        '''
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
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
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (1, n_timepoints, n_features)
            Labels of data points
        '''
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        validate_data_3d(X)
        check_one_sample(X)
        check_if_fitted(self)

        scores = self.predict_anomaly_scores(X)
        return scores > self._threshold

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            Array of shape (1, n_timepoints, n_features)
            Anomaly scores.
        '''
        validate_data_3d(X)
        check_one_sample(X)
        check_if_fitted(self)

        return np.abs((X - self._mean) / self._std)


class WindowingDetector(BaseDetector):
    '''
    WindowingDetector allows for regular outlier/anomaly detection algorithms
    to be used on time series data. Subsequences are extracted from the
    original time series and then are served as vectors to the regular models.

    Parameters
    ----------
    detector
        Detector model to be applied on subsequences.
    window_size: int
        Size of the subsequences.
    standardize: bool
        Whether the subsequences should be standardized or not.
    scaler_kwargs: dictionary
        Dictionary of MeanVarianceScaler parameters
    '''

    def __init__(
        self,
        detector,
        window_size: int,
        standardize: bool = False,
        **scaler_kwargs,
    ) -> None:
        if window_size < 1:
            raise ValueError('Window size should be > 0')
        self._fitted = False
        self._window_size = window_size
        self._detector = detector
        self._standardize = standardize
        self._scaler = None
        if self._standardize:
            self._scaler = MeanVarianceScaler(**scaler_kwargs)

    def fit(self, X: np.ndarray):
        '''
        Fit supplied model to transformed data.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self
        '''
        validate_data_3d(X)
        check_one_sample(X)
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        self._fitted = True
        X_sequences = self._prep(X)
        if self._standardize:
            X_sequences = self._scaler.fit_transform(X_sequences)

        self._detector.fit(X_sequences)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict anomaly labels

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Input data

        Returns
        -------
        np.ndarray
            _description_
        '''
        check_if_fitted(self)
        validate_data_3d(X)
        check_one_sample(X)
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        X_sequences = self._prep(X)
        if self._standardize:
            X_sequences = self._scaler.transform(X_sequences)

        labels = self._detector.predict(X_sequences)
        return labels

    def predict_anomaly_scores(self, X: np.ndarray):
        check_if_fitted(self)
        validate_data_3d(X)
        check_one_sample(X)
        X_sequences = self._prep(X)
        if self._standardize:
            X_sequences = self._scaler.transform(X_sequences)

        return reduce_window_scores(
            self._detector.predict_anomaly_scores(X_sequences),
            self._window_size,
        )

    def _prep(self, X: np.ndarray):
        return sliding_window_sequences(X, window_size=self._window_size)


class KNN(BaseDetector):
    def __init__(self, **kwargs) -> None:
        self._fitted = False
        self._detector = NearestNeighbors(**kwargs)

    def fit(self, X: np.ndarray):
        '''
        Fit the nearest neighbor estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints)
            Training data.

        Returns
        -------
        self
            Fitted KNN estimator.
        '''
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_2d(X)
        self._fitted = True
        self._detector.fit(X)
        dist, _ = self._detector.kneighbors(X, self._detector.n_neighbors + 1)
        self._threshold = np.max(np.mean(dist[:, 1:], axis=1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, 1)
            Anomaly labels
        '''
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        dist = self.predict_anomaly_scores(X)
        return dist > self._threshold

    def predict_anomaly_scores(self, X: np.ndarray):
        '''
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints)
            Data

        Returns
        -------
        np.ndarray, np.ndarray
            np.ndarray of shape (n_samples, 1)
            Distance to nearest data point for each sample
            np.ndarray of shape (n_samples, 1)
            Index of nearest neighbor in training data for each sample from X

        '''
        check_if_fitted(self)
        dist, _ = self._detector.kneighbors(X, self._detector.n_neighbors)
        return np.mean(dist[:, 1:], axis=1)
