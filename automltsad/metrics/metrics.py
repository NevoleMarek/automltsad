import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve


def point_adjust(
    labels: np.ndarray, scores: np.ndarray, threshold: float, k: float = 0
):
    '''
    Code from: https://github.com/tuslkkk/tadpak/blob/master/tadpak/pak.py

    The implementation of PA%K evaluation protocol of Towards a Rigorous
    Evaluation of Time-sereis anomaly detection, Siwon Kim, Kukjin Choi,
    Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon, AAAI 2022.

    This function point adjusts predictions.

    Assumption is that in real scenario it's sufficient to detect just part of
    the anomaly, not the entire sequence.

    If just k% of anomaly sequence is detected as anomaly count the entire
    sequence as True positive.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    scores : np.ndarray
        Anomaly scores
    threshold : float
        Anomaly threshold
    k : float, optional
        Value in [0,1] , by default 0

    Returns
    -------
    np.ndarray
        Point adjusted predictions
    '''

    predicts = scores > threshold

    anml_seg_starts = np.where(np.diff(labels, prepend=0) == 1)[0]
    anml_seg_ends = np.where(np.diff(labels, prepend=0) == -1)[0]

    # case where labels end with anomaly (labels[end] == 1)
    if len(anml_seg_starts) == len(anml_seg_ends) + 1:
        anml_seg_ends = np.append(anml_seg_ends, len(predicts))

    for i in range(len(anml_seg_starts)):
        if predicts[anml_seg_starts[i] : anml_seg_ends[i]].sum() > k * (
            anml_seg_ends[i] - anml_seg_starts[i]
        ):
            predicts[anml_seg_starts[i] : anml_seg_ends[i]] = 1
    return predicts


def f1_pa(
    labels: np.ndarray, scores: np.ndarray, threshold: float, k: float = 0
):
    '''
    F1 score with point adjusted protocol @ k

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    scores : np.ndarray
        Anomaly scores
    threshold : float
        Anomaly threshold
    k : float, optional
        Value in [0,1] , by default 0

    Returns
    -------
    float
        F1 score on point adjusted predictions
    '''
    preds = point_adjust(labels, scores, threshold, k)
    return f1_score(labels, preds)


def f1_pa_curve(labels: np.ndarray, scores: np.ndarray, threshold: float):
    '''
    Curve for f1 score at different k
    for point adjusted predictions.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    pred : np.ndarray
        Predicted labels

    Returns
    -------
    np.ndarray
        Different k values
    np.ndarray
        F1 score @ k values
    '''
    f1_pa_k = np.zeros((11,))
    ks = np.zeros((11,))
    for i, k in enumerate(np.arange(0, 1.1, 0.1)):
        f1_pa_k[i] = f1_pa(labels, scores, threshold, k)
        ks[i] = k
    return ks, f1_pa_k


def f1_pa_auc_score(labels: np.ndarray, scores: np.ndarray, threshold: float):
    '''
    Area under curve for f1 score at different k
    for point adjusted predictions.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    pred : np.ndarray
        Predicted labels

    Returns
    -------
    float
        Area under curve in [0,1] range
    '''
    k, f1 = f1_pa_curve(labels, scores, threshold)
    return auc(k, f1)
