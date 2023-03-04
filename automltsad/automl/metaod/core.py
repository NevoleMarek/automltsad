import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-1 * a * x))


def sigmoid_derivate(x, a=1):
    return sigmoid(x, a) * (1 - sigmoid(x, a))


class MetaODClass(object):
    def __init__(
        self,
        train_performance,
        valid_performance,
        n_factors=40,
        learning='sgd',
        verbose=False,
    ):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        train_performance matrix which is ~ user x item

        Params
        ======
        train_performance : (ndarray)
            User x Item matrix with corresponding train_performance
        n_factors : (int)
            Number of latent factors to use in matrix
            factorization model
        learning : (str)
            Method of optimization. Options include
            'sgd' or 'als'.
        verbose : (bool)
            Whether or not to printout training progress
        """

        self.ratings = train_performance
        self.valid_ratings = valid_performance
        self.n_users, self.n_items = train_performance.shape
        self.n_factors = n_factors
        self.learning = learning
        if self.learning == 'sgd':
            self.n_samples, self.n_models = (
                self.ratings.shape[0],
                self.ratings.shape[1],
            )
        self._v = verbose
        self.train_loss_ = [0]
        self.valid_loss_ = [0]
        self.scalar_ = None
        self.pca_ = None

    def get_train_dcg(self, user_vecs, item_vecs):
        # make sure it is non zero
        user_vecs[np.isnan(self.user_vecs)] = 0

        ndcg_s = []
        for w in range(self.ratings.shape[0]):
            ndcg_s.append(
                ndcg_score(
                    [self.ratings[w, :]],
                    [np.dot(user_vecs[w, :], item_vecs.T)],
                )
            )

        return np.mean(ndcg_s)

    def train(
        self,
        meta_features,
        valid_meta=None,
        n_iter=10,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=10,
    ):
        """Train model for n_iter iterations from scratch."""

        self.pca_ = PCA(n_components=self.n_factors)
        self.pca_.fit(meta_features)

        meta_features_pca = self.pca_.transform(meta_features)
        meta_valid_pca = self.pca_.transform(valid_meta)

        self.scalar_ = StandardScaler()
        self.scalar_.fit(meta_features_pca)

        meta_features_scaled = self.scalar_.transform(meta_features_pca)
        meta_valid_scaled = self.scalar_.transform(meta_valid_pca)

        self.user_vecs = meta_features_scaled

        self.item_vecs = np.random.normal(
            scale=1.0 / self.n_factors, size=(self.n_items, self.n_factors)
        )

        ctr = 1
        np_ctr = 1
        while ctr <= n_iter:
            self.regr_multirf = MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, n_jobs=4
                )
            )

            # make sure it is non zero
            self.user_vecs[np.isnan(self.user_vecs)] = 0

            self.regr_multirf.fit(meta_features_scaled, self.user_vecs)

            meta_valid_scaled_new = self.regr_multirf.predict(
                meta_valid_scaled
            )

            ndcg_s = []
            for w in range(self.ratings.shape[0]):
                ndcg_s.append(
                    ndcg_score(
                        [self.ratings[w, :]],
                        [np.dot(self.user_vecs[w, :], self.item_vecs.T)],
                        k=self.n_items,
                    )
                )

            self.train_loss_.append(np.mean(ndcg_s))

            ndcg_s = []
            for w in range(self.valid_ratings.shape[0]):
                ndcg_s.append(
                    ndcg_score(
                        [self.valid_ratings[w, :]],
                        [
                            np.dot(
                                meta_valid_scaled_new[w, :], self.item_vecs.T
                            )
                        ],
                        k=self.n_items,
                    )
                )
            self.valid_loss_.append(np.mean(ndcg_s))

            print(
                'MetaOD',
                ctr,
                'train',
                self.train_loss_[-1],
                'valid',
                self.valid_loss_[-1],
                'learning rate',
                learning_rate,
            )

            # improvement is smaller than 1 perc
            if (
                (self.valid_loss_[-1] - self.valid_loss_[-2])
                / self.valid_loss_[-2]
            ) <= 0.001:
                np_ctr += 1
            else:
                np_ctr = 1
            if np_ctr > 5:
                break

            train_indices = list(range(self.n_samples))
            np.random.shuffle(train_indices)

            for h in train_indices:

                uh = self.user_vecs[h, :].reshape(1, -1)
                grads = []

                for i in range(self.n_models):
                    # outler loop
                    vi = self.item_vecs[i, :].reshape(-1, 1)
                    phis = []
                    rights = []
                    rights_v = []

                    js = list(range(self.n_models))
                    js.remove(i)

                    for j in js:
                        vj = self.item_vecs[j, :].reshape(-1, 1)

                        temp_vt = sigmoid(
                            np.ndarray.item(np.matmul(uh, (vj - vi))), a=1
                        )
                        temp_vt_derivative = sigmoid_derivate(
                            np.ndarray.item(np.matmul(uh, (vj - vi))), a=1
                        )

                        phis.append(temp_vt)
                        rights.append(temp_vt_derivative * (vj - vi))
                        rights_v.append(temp_vt_derivative * uh)
                    phi = np.sum(phis) + 1.5
                    rights = np.asarray(rights).reshape(
                        self.n_models - 1, self.n_factors
                    )
                    rights_v = np.asarray(rights_v).reshape(
                        self.n_models - 1, self.n_factors
                    )

                    right = np.sum(np.asarray(rights), axis=0)
                    right_v = np.sum(np.asarray(rights_v), axis=0)

                    grad = (
                        (10 ** (self.ratings[h, i]) - 1)
                        / (phi * (np.log(phi)) ** 2)
                        * right
                    )
                    grad_v = (
                        (10 ** (self.ratings[h, i]) - 1)
                        / (phi * (np.log(phi)) ** 2)
                        * right_v
                    )

                    self.item_vecs[i, :] += learning_rate * grad_v

                    grads.append(grad)

                grads_uh = np.asarray(grads)
                grad_uh = np.sum(grads_uh, axis=0)

                self.user_vecs[h, :] -= learning_rate * grad_uh

            ctr += 1

        self.ratings = None
        self.valid_ratings = None
        return self

    def predict(self, test_meta):

        test_meta = check_array(test_meta)
        assert test_meta.shape[1] == 200

        test_meta_scaled = self.pca_.transform(test_meta)
        # print('B', test_meta_scaled.shape)

        test_meta_scaled = self.scalar_.transform(test_meta_scaled)
        test_meta_scaled = self.regr_multirf.predict(test_meta_scaled)

        # predicted_scores = np.dot(test_k, self.item_vecs.T) + self.item_bias
        predicted_scores = np.dot(test_meta_scaled, self.item_vecs.T)
        # print(predicted_scores.shape)
        assert predicted_scores.shape[0] == test_meta.shape[0]
        assert predicted_scores.shape[1] == self.n_models

        return predicted_scores
