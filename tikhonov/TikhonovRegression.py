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
from sklearn.linear_model import Ridge
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
    :param gamma: array-like, shape = [n_features, n_features].
        Covariance matrix of the prior

    :return:
    beta_hat: beta weight estimates
    """
    if sigma is None:
        sigma = np.eye(x.shape[1])
    return np.dot(np.linalg.inv(np.dot(x.T, x) +
                  np.linalg.inv(sigma) *
                  alpha), np.dot(x.T, y))


def find_gamma(x):
    """
    Cholesky decomposition if covariance matrix is given
    :param x: feature x feature covariance matrix. This is used to
    find a Tikhonov matrix L such that:
     inv(X.T * X) = L.T * L
    :return: L: the Tikhonov matrix for this situation
    """
    if not np.all(x.T == x):
        raise ValueError("Input matrix is not symmetric. "
                         "Are you sure it is covariance?")
    return np.linalg.cholesky(np.linalg.pinv(x)).T
    # _, s, vh = np.linalg.svd(x-x.mean(0), full_matrices=False)
    # return np.dot(np.diag(1/s[s > cutoff]), vh[s > cutoff, :])


def _standarize_params(x, gamma):
    """
    Calculates parameters associated with rotating the data to standard form
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data    b = inv(X.T*X + inv(Sigma)*alpha) * X.T * y
    :param gamma: array-like, shape = [n_features, n_regularizers].
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
    kp, ko, rp = _qr(gamma.T)
    if ko.shape is ():  # there is no lower part of matrix
        ho, hq, to = np.array(1.0), np.array(1.0), np.array(1.0)
    else:
        ho, hq, to = _qr(np.dot(x, ko))
    if hq.shape is ():  # special case where L is square
                        # (saves computational time later
        ko, to, ho = None, None, None
    return hq, kp, rp, ko, ho, to


def to_standard_form(x, y, gamma):
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
    :param gamma: array-like, shape = [n_features, n_regularizers]
        Generally, L.T * L is the inverse covariance matrix of the data.

    :return:
    x_new : {array-like}, transformed x
    y_new : {array-like}, transformed y
    """
    hq, kp, rp, _, _, _ = _standarize_params(x, gamma)
    # this is derived by doing a bit of algebra:
    # x_new = hq.T * x * kp * inv(rp).T
    x_new = solve_triangular(rp, np.dot(kp.T, np.dot(x.T, hq))).T
    y_new = np.dot(hq.T, y)
    return x_new, y_new


def to_general_form(b, x, y, gamma):
    """
    Converts weights back into general form space.
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param b: array-like, shape = [n_features] or [n_features, n_targets]
        regression coefficients
    :param gamma: arra-like, shape = [n_features, n_regularizers]
        Generally, gamma.T* gamma is the inverse covariance matrix of the data
    :return:
    b : ridge coefficients rotated back to original space
    """
    hq, kp, rp, ko, ho, to = _standarize_params(x, gamma)

    if ko is to is ho is None:
        gamma_inv = np.dot(kp, np.linalg.pinv(rp.T))
        return np.dot(gamma_inv, b)
        # return np.linalg.solve(np.dot(rp.T, kp.T), b)
    else:
        gamma_inv = np.dot(kp, np.linalg.inv(rp.T))
        kth = np.dot(ko, np.dot(np.linalg.inv(to, ho.T)))
        resid = y - np.dot(x, np.dot(gamma_inv, b))
        # kth and resid should be 0...
        return np.dot(gamma_inv, b) + np.dot(kth, resid)


def fit_learner(x, y, gamma, ridge=None):
    """
    Returns an trained model that works exactly the same as Ridge,
    but fit optimally
    """
    if ridge is None:
        ridge = Ridge()
    x_new, y_new = to_standard_form(x, y, gamma)
    ta_est_standard = ridge.fit(x_new, y_new).coef_
    ta_est = to_general_form(ta_est_standard, x, y, gamma)
    ridge.coef_ = ta_est
    return ridge
