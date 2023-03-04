# -*- coding: utf-8 -*-
import numpy as np


def Diff(li1, li2):
    return list(set(li1) - set(li2))


def argmaxatn(w, nth):
    w = np.asarray(w).ravel()
    t = np.argsort(w)
    return t[-1 * nth]


def fix_nan(X):
    # TODO: should store the mean of the meta features to be used for test_meta
    # replace by 0 for now
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = np.nan_to_num(X)
    return X
