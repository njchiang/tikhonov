"""
Tikhonov regression

L2-regularized regression using a non-diagonal regularization matrix using (Truncated) SVD
"""

# Author: Jeff Chiang <jeff.njchiang@gmail.com>
# License: BSD 3 clause

# import time
# import logging

import numpy as np
from scipy import linalg


# TODO : use fit-transform to return the SKL version for just the rotation...

# FunctionTransformer: func = to_standard, inverse_func = to_original
# diedrichsen method
# should be run on betas, not raw signal
def _compute_g(u):
    return np.dot(u, u.T)/u.shape[1]


def _rotate_m(m, u):
    u, s, _ = np.linalg.svd(u/np.sqrt(u.shape[1]))
    return np.dot(u, np.diag(s))


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


# TODO : estimate datas, and take covariance matrix.
def _check_x_gamma(x, gamma):
    m, n1 = x.shape
    p, n2 = gamma.shape
    if p > n2:
        raise ValueError("Regularization matrix is rank deficient. "
                         "Gamma does not satisfy p < n. "
                         "p: %d | n: %d" % (p, n2))
    if n1 != n2:
        raise ValueError("Number of features in data and "
                         "regularization matrix do not "
                         "correspond: %d != %d"
                         % (n1, n2))
    if m + p < n1:
        raise ValueError("Number of data + regularization samples "
                         "is not greater than features. M+P !> N "
                         "m: %d | p: %d | n: %d" % (m, p, n1))


def analytic_tikhonov(x, y, alpha, gamma):
    return np.dot(np.linalg.inv(np.dot(x.T, x) +
                                np.linalg.inv(gamma) *
                                alpha), np.dot(x.T, y))


def find_gamma(x, cutoff=1e-14):
    ## TODO: fix
    """
    Cholesky decomposition if covariance matrix is given...
    :param x: matrix of same axis 1 as data (# of features). This is used to find a Tikhonov matrix L such that:
     inv(X.T * X) = L.T * L
    :return: L: the Tikhonov matrix for this situation
    """
    _, s, vh = np.linalg.svd(x-x.mean(0), full_matrices=False)
    return np.dot(np.diag(1/s[s > cutoff]), vh[s > cutoff, :])


# ONLY IF GAMMA IS KNOWN
def to_standard_form(x, y, gamma):
    """
    Converts x and y into "standard form" in order to efficiently solve the Tikhonov regression problem.
    L is the Tikhonov regularizer, such that L.T * L can be the inverse covariance matrix of the data.
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data

    y :
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param gamma: array-like, shape = [n_features, n_regularizers]
        Generally, L.T * L is the inverse covariance matrix of the data.

    :return:
    x_new : {array-like}, transformed x
    y_new : {array-like}, transformed y
    L_inv : {array-like}, pseudoinverse of L
    ko : {array-like or 1}
    to : {array-like or 1}
    ho : {array-like or 1}
    """
    kp, ko, rp = _qr(gamma.T)
    if ko.shape is ():  # there is no lower part of matrix
        ho, hq, to = np.array(1.0), np.array(1.0), np.array(1.0)
    else:
        ho, hq, to = _qr(np.dot(x, ko))
    # there must be a better implementation for rp_inv...
    # one way to do it
    # slow way:
    # rp_inv = np.linalg.pinv(rp)
    # L_inv = np.dot(kp, rp_inv.T)
    # x_new = reduce(np.dot, [hq.T, x, L_inv])
    # solving triangular
    x_new = linalg.solve_triangular(rp, np.dot(kp.T, np.dot(x.T, hq))).T
    y_new = np.dot(hq.T, y)
    if hq.shape is ():  # special case where L is square (saves computational time later
        ko, to, ho = None, None, None
    return x_new, y_new, hq, kp, rp, ko, ho, to


def to_general_form(x, y, b, kp, rp, ko=None, to=None, ho=None):
    """
    Converts weights back into general form space.
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param b: array-like, shape = [n_features] or [n_features, n_targets]
        regression coefficients
    :param kp: array-like, shape = [n_samples] or [n_samples, n_targets]
        first block matrix of QR factorization of L.T. kp * rp^-1.T is inv(L)
    :param rp: array-like, shape = [n_samples] or [n_samples, n_targets]
        upper triangular matrix of QR factorization of L.T
    :param ko: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param to: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param ho: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values

    :return:
    b : ridge coefficient rotated back to original space
    """
    # this should work, but for some reason it is not 0 when L is square.
    # hot fix for now.
    if ko is to is ho is None:
        gamma_inv = np.dot(kp, np.linalg.inv(rp.T))
        return np.dot(gamma_inv, b)
        # return np.linalg.solve(np.dot(rp.T, kp.T), b)
    else:
        gamma_inv = np.dot(kp, np.linalg.inv(rp.T))
        kth = np.dot(ko, np.dot(np.linalg.inv(to, ho.T)))
        resid = y - np.dot(x, np.dot(gamma_inv, b))
        # kth and resid should be 0...
        return np.dot(gamma_inv, b) + np.dot(kth, resid)
