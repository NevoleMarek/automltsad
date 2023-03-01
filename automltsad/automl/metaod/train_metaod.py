# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:27 2020

@author: yuezh
"""

import os
import random

import numpy as np
import pandas as pd
from joblib import dump
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from automltsad.automl.metaod.models.core import MetaODClass
from automltsad.automl.metaod.models.gen_meta_features import generate_meta_features
from automltsad.automl.metaod.models.utility import fix_nan, read_arff

# get statistics of the training data
n_datasets, n_configs = roc_mat_red.shape[0], roc_mat_red.shape[1]
data_headers = roc_mat[2:, 0]
config_headers = roc_df.columns[4:]
dump(config_headers, 'model_list.joblib')

# Load datasets and generate meta features
meta_mat = np.zeros([n_datasets, 200])
X = check_array(X).astype('float64')
meta_mat[j, :], meta_vec_names = generate_meta_features(X)

# use cleaned and transformed meta-features
meta_scalar = MinMaxScaler()
meta_mat_transformed = meta_scalar.fit_transform(meta_mat)
meta_mat_transformed = fix_nan(meta_mat_transformed)
dump(meta_scalar, 'meta_scalar.joblib')
# %% train model

# split data into train and valid

full_list = list(range(n_datasets))

train_index = full_list[:n_train]
valid_index = full_list[n_train:]

train_set = roc_mat_red[train_index, :].astype('float64')
valid_set = roc_mat_red[valid_index, :].astype('float64')

train_meta = meta_mat_transformed[train_index, :].astype('float64')
valid_meta = meta_mat_transformed[valid_index, :].astype('float64')

clf = MetaODClass(
    train_set, valid_performance=valid_set, n_factors=30, learning='sgd'
)
clf.train(
    n_iter=50,
    meta_features=train_meta,
    valid_meta=valid_meta,
    learning_rate=0.05,
    max_rate=0.9,
    min_rate=0.1,
    discount=1,
    n_steps=8,
)

# U = clf.user_vecs
# V = clf.item_vecs

# # # print(EMF.regr_multirf.predict(test_meta).shape)
# predicted_scores = clf.predict(valid_meta)
# predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
# print()
# output transformer (for meta-feature) and the trained clf
dump(clf, 'train_' + str(seed) + '.joblib')

#%%
# # %%
# import pickle
# from metaod.models.core import MetaODClass

# if __name__ == "__main__":
#     # # code for standalone use
#     # t = Thing("foo")
#     # Thing.__module__ = "thing"
#     # t.save("foo.pickle")
#     # MetaODClass.__module__ = "metaod"
#     file = open('test.pk', 'wb')
#     pickle.dump(clf, file)

# # # file = open('rf.pk', 'wb')
# # # pickle.dump(clf.user_vecs, file)
