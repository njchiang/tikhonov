"""
Tikhonov regression,

based on implementation by Stout and Kalivas, 2006. Journal of Chemometrics
L2-regularized regression using a non-diagonal regularization matrix

This can be done in two ways, by setting the original problem into
"standard space", such that regular ridge regression can be employed,
or solving the equation in original space. As number features increases,
rotating the original problem should be faster
"""
# Author: Jeff Chiang <jeff.njchiang@gmail.com>
# License: BSD 3 clause
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.utils import check_X_y
# from sklearn.preprocessing import FunctionTransformer


# QR method
def _qr(x):
    m, n = x.shape
    q, r = np.linalg.qr(x, mode='complete')
    rp = r[:n, :]
    qp = q[:, :n]
    qo = q[:, n:]
    # check for degenerate case
    if qo.shape[1] == 0:
        qo = np.array(1.0)
    return qp, qo, rp


def analytic_tikhonov(x, y, alpha, sigma=None):
    """
    Solves tikhonov regularization problem with the covariance of the
    weights as a prior
    b = inv(X.T*X + inv(Sigma)*alpha) * X.T * y
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param alpha: scalar, regularization parameter
    :param sigma: array-like, shape = [n_features, n_features].
        Covariance matrix of the prior


    :return:
    beta_hat: beta weight estimates
    """
    if sigma is None:
        sigma = np.eye(x.shape[1])
    return np.dot(np.linalg.pinv(np.dot(x.T, x) +
                  np.linalg.pinv(sigma) *
                  alpha), np.dot(x.T, y))


def find_tikhonov_from_covariance(x, cutoff=.0001, eps=1e-10):
    """
    Use truncated-SVD to find Tikhonov matrix
    :param x: feature x feature covariance matrix. This is used to
    find a Tikhonov matrix L such that:
     inv(x) = L.T * L
    :param cutoff: cutoff value for singular value magnitude. if it's too low,
    rank will suffer.
    :return: L: the Tikhonov matrix for this situation
    """
    if not np.allclose(x.T, x):
        raise ValueError("Input matrix is not symmetric. "
                         "Are you sure it is covariance?")
    _, s, vt = np.linalg.svd(np.linalg.pinv(x))
    return np.dot(np.diag(np.sqrt(s[s > cutoff])), vt[s > cutoff])
    # return np.linalg.cholesky(np.linalg.pinv(x)).T
    # _, s, vh = np.linalg.svd(x-x.mean(0), full_matrices=False)
    # return np.dot(np.diag(1/s[s > cutoff]), vh[s > cutoff, :])


def _standardize_params(x, L):
    """
    Calculates parameters associated with rotating the data to standard form
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data    b = inv(X.T*X + inv(Sigma)*alpha) * X.T * y
    :param L: array-like, shape = [n_features, n_regularizers].
        Tikhonov matrix
    returns:
        hq: array-like
        kp: array-like, shape = [n_samples] or [n_samples, n_targets]
            first block matrix of QR factorization of L.T.
            kp * rp^-1.T is inv(L)
        rp: array-like, shape = [n_samples] or [n_samples, n_targets]
            upper triangular matrix of QR factorization of L.T
        ko: array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values
        to: array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values
        ho: array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values
    """
    kp, ko, rp = _qr(L.T)
    if ko.shape is ():  # there is no lower part of matrix
        ho, hq, to = np.array(1.0), np.array(1.0), np.array(1.0)
    else:
        ho, hq, to = _qr(np.dot(x, ko))
    if hq.shape is ():  # special case where L is square
                        # (saves computational time later
        ko, to, ho = None, None, None
    return hq, kp, rp, ko, ho, to


def to_standard_form(x, y, L):
    """
    Converts x and y into "standard form" in order to efficiently
    solve the Tikhonov regression problem.
    gamma is the Tikhonov regularizer, such that L.T * L can be the inverse
    covariance matrix of the data.
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param L: array-like, shape = [n_features, n_regularizers]
        Generally, L.T * L is the inverse covariance matrix of the data.

    :return:
    x_new : {array-like}, transformed x
    y_new : {array-like}, transformed y
    """
    hq, kp, rp, _, _, _ = _standardize_params(x, L)
    # this is derived by doing a bit of algebra:
    # x_new = hq.T * x * kp * inv(rp).T
    x_new = solve_triangular(rp, np.dot(kp.T, np.dot(x.T, hq))).T
    y_new = np.dot(hq.T, y)
    return x_new, y_new


def to_general_form(b, x, y, L):
    """
    Converts weights back into general form space.
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param b: array-like, shape = [n_features] or [n_features, n_targets]
        regression coefficients
    :param L: arra-like, shape = [n_features, n_regularizers]
        Generally, L.T* L is the inverse covariance matrix of the data
    :return:
    b : ridge coefficients rotated back to original space
    """
    hq, kp, rp, ko, ho, to = _standardize_params(x, L)

    if ko is to is ho is None:
        L_inv = np.dot(kp, np.linalg.pinv(rp.T))
        return np.dot(L_inv, b)
        # return np.linalg.solve(np.dot(rp.T, kp.T), b)
    else:
        L_inv = np.dot(kp, np.linalg.pinv(rp.T))
        kth = np.dot(ko, np.dot(np.linalg.pinv(to), ho.T))
        resid = y - np.dot(x, np.dot(L_inv, b))
        # kth and resid should be 0...
        return np.dot(L_inv, b) + np.dot(kth, resid)


def fit_learner(x, y, L, ridge=None):
    """
    Returns an trained model that works exactly the same as Ridge,
    but fit optimally
    """
    if ridge is None:
        ridge = Ridge(fit_intercept=False)
    x_new, y_new = to_standard_form(x, y, L)
    ta_est_standard = ridge.fit(x_new, y_new).coef_
    ta_est = to_general_form(ta_est_standard.T, x, y, L)
    ridge.coef_ = ta_est.T
    return ridge


class Tikhonov(Ridge):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
        super(Ridge, self).__init__(alpha=alpha, fit_intercept=fit_intercept,
                                    normalize=normalize, copy_X=copy_X,
                                    max_iter=max_iter, tol=tol, solver=solver,
                                    random_state=random_state)

    def fit(self, X, y, L=None, sample_weight=None):
        if self.solver in ('sag', 'saga'):
            _dtype = np.float64
        else:
            # all other solvers work at both float precision levels
            _dtype = [np.float64, np.float32]

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=_dtype,
                         multi_output=True, y_numeric=True)

        if ((sample_weight is not None) and
                    np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if L is not None:
            x_new, y_new = to_standard_form(X, y, L)
            model = super(Ridge, self)
            model.fit(x_new, y_new, sample_weight=sample_weight)
            standard_coefs = to_general_form(self.coef_.T, X, y, L)
            self.coef_ = standard_coefs.T
        else:
            super(Ridge, self).fit(X, y, sample_weight=sample_weight)

        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class TikhonovCV(RidgeCV):

    def __init__(self, alphas=(0.1, 1.0, 10.0),
                 fit_intercept=True, normalize=False, scoring=None, copy_X=True,
                 cv=None, gcv_mode=None,
                 store_cv_values=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.copy_X = copy_X

    def fit(self, X, y, L=None, sample_weight=None):

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64,
                         multi_output=True, y_numeric=True)

        if ((sample_weight is not None) and
                    np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if L is not None:
            x_new, y_new = to_standard_form(X, y, L)
            model = super(RidgeCV, self)
            model.fit(x_new, y_new, sample_weight=sample_weight)
            standard_coefs = to_general_form(self.coef_.T, X, y, L)
            self.coef_ = standard_coefs.T
        else:
            super(RidgeCV, self).fit(X, y, sample_weight=sample_weight)

        self._set_intercept(X_offset, y_offset, X_scale)
        return self