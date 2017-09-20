"""
Tikhonov regression

L2-regularized regression using a non-diagonal regularization matrix using (Truncated) SVD
"""

# Author: Jeff Chiang <jeff.njchiang@gmail.com>
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
import warnings
# import time
# import logging

import numpy as np
from scipy import linalg
from sklearn.utils import check_consistent_length
from sklearn.utils import check_array

# TODO : use fit-transform to return the SKL version for just the rotation...

# FunctionTransformer: func = to_standard, inverse_func = to_original
# Any representational model can
#  be brought into this diagonal form by setting the columns of M to the eigenvectors of G, each
# one multiplied by the square root of the corresponding eigenvalue:
# Or Generalized Cross validation to estimate matrix

def _qr(L):
    m, n = L.shape
    q, r = np.linalg.qr(L, mode='complete')
    rp = r[:n, :]
    qp = q[:, :n]
    qo = q[:, n:]
    # check for degenerate case
    if qo.shape[1] == 0:
        qo = np.array(1.0)
    return qp, qo, rp


def _check_alphas(alpha, n_targets):
    # There should be either 1 or n_targets penalties
    alpha = np.asarray(alpha).ravel()
    if alpha.size not in [1, n_targets]:
        raise ValueError("Number of targets and number of penalties "
                         "do not correspond: %d != %d"
                         % (alpha.size, n_targets))

    if alpha.size == 1 and n_targets > 1:
        alpha = np.repeat(alpha, n_targets)

    return alpha

# TODO : estimate datas, and take covariance matrix.
def _check_x_gamma(x, L):
    m, n1 = x.shape
    p, n2 = L.shape
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


def find_gamma(x, cutoff=1e-14):
    ## TODO: fix
    """
    :param x: matrix of same axis 1 as data (# of features). This is used to find a Tikhonov matrix L such that:
     inv(X.T * X) = L.T * L
    :return: L: the Tikhonov matrix for this situation
    """
    _, s, vh = np.linalg.svd(x-x.mean(0), full_matrices=False)
    return np.dot(np.diag(1/s[s > cutoff]), vh[s > cutoff, :])


def to_standard_form(x, y, L):
    """
    Converts x and y into "standard form" in order to efficiently solve the Tikhonov regression problem.
    L is the Tikhonov regularizer, such that L.T * L can be the inverse covariance matrix of the data.
    :param x: {array-like},
        shape = [n_samples, n_features]
        Training data

    y :
    :param y: array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values
    :param L: array-like, shape = [n_features, n_regularizers]
        Generally, L.T * L is the inverse covariance matrix of the data.

    :return:
    x_new : {array-like}, transformed x
    y_new : {array-like}, transformed y
    L_inv : {array-like}, pseudoinverse of L
    ko : {array-like or 1}
    to : {array-like or 1}
    ho : {array-like or 1}
    """
    kp, ko, rp = _qr(L.T)
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
    x_new = linalg.solve_triangular(rp, reduce(np.dot, [kp.T, x.T, hq])).T
    y_new = np.dot(hq.T, y)
    if hq.shape is (): # special case where L is square (saves computational time later
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
        Li = np.dot(kp, np.linalg.inv(rp.T))
        return np.dot(Li, b)
        # return np.linalg.solve(np.dot(rp.T, kp.T), b)
    else:
        Li = np.dot(kp, np.linalg.inv(rp.T))
        kth = reduce(np.dot, [ko, np.linalg.inv(to), ho.T])
        resid = y - reduce(np.dot, [x, Li, b])
        # kth and resid should be 0...
        return np.dot(Li, b) + np.dot(kth, resid)
