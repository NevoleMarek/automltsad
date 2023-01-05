import logging

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from automltsad.detectors import BaseDetector
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

    def fit(self, X: np.ndarray, y=None):
        '''
        Fit the trivial detector on training dataset.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Training data.
        y : ignored

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

    def fit(self, X: np.ndarray, y=None):
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
    '''
    Nearest neighbor detector.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for kneighbors queries.
    radius : float, default=1.0
        Range of parameter space to use by default for radius_neighbors
        queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree'
        - 'kd_tree'
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to fit method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.
        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
    '''

    def __init__(
        self,
        n_neighbors=5,
        radius=1.0,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        n_jobs=None,
    ) -> None:
        self._fitted = False
        self._detector = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

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


class IsolationForestAD(BaseDetector):
    '''
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm
    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.
    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max(1, int(max_features * n_features_in_))` features.
        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longerr runtime.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    n_jobs : int, default=None
        The number of jobs to run in parallel
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
    verbose : int, default=0
        Controls the verbosity of the tree building process.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    '''

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ) -> None:
        self._fitted = False
        self._detector = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

    def fit(self, X: np.ndarray):
        '''
        Fit the isolation forest estimator.

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
        return self._detector.predict(X)

    def predict_anomaly_scores(self, X: np.ndarray):
        '''
        Predict anomaly scores.

        The higher the score the more anomalous the point.

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
        return -self._detector.score_samples(X)
