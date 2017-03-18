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
from scipy import sparse
# import scipy.stats as sp
from scipy.sparse import linalg as sp_linalg
from sklearn.linear_model.base import LinearClassifierMixin, LinearModel, _rescale_data
from sklearn.base import RegressorMixin
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.utils import compute_sample_weight
from sklearn.utils import check_consistent_length
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_X_y
from sklearn.utils import check_array
# from sklearn.utils import column_or_1d
from sklearn.model_selection import GridSearchCV
from sklearn.externals import six
from sklearn.metrics.scorer import check_scoring
# from sklearn.linear_model.sag import sag_solver
from sklearn.linear_model.ridge import ridge_regression


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


def _svd_and_cache(x, y, singcutoff=None):
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    # in case we want to truncate tiny singular values
    if singcutoff is not None:
        nGood = np.sum(s > singcutoff)
        u = u[:, :nGood]
        s = s[:nGood]
        vh = vh[:nGood]
    ur = np.dot(u.T, np.nan_to_num(y))
    return u, s, vh, ur


def _solve_cache_svd(vh, s, alphas, ur, l=None):
    # alphas is assumed to be an array of the correct size
    wt = np.zeros((vh.shape[1], ur.shape[1]))
    for a in np.unique(alphas):  # in case there are many unique alphas
        selvox = np.nonzero(alphas == a)[0]
        awt = reduce(np.dot, [vh.T, np.diag(s / (s ** 2 + a )), ur[:, selvox]])
        wt[:, selvox] = awt
    # to follow SKL convention
    return wt.T


def _check_data(X, y, solver=None):
    if solver == 'sag':
        X = check_array(X, accept_sparse=['csr'],
                        dtype=np.float64, order='C')
        y = check_array(y, dtype=np.float64, ensure_2d=False, order='F')
    else:
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64)
        y = check_array(y, dtype='numeric', ensure_2d=False)

    check_consistent_length(X, y)

    n_samples, n_features = X.shape

    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True

    n_samples_, n_targets = y.shape

    if n_samples != n_samples_:
        raise ValueError("Number of samples in X and y does not correspond:"
                         " %d != %d" % (n_samples, n_samples_))
    return X, y, n_samples, n_features, n_targets, ravel


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


def cache_ridge_regression(X, y, cache=None, alpha=1.0, sample_weight=None):
    """Solve the ridge equation by the method of normal equations.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------

    cache : (tuple)
        shape = (4)
        Contains SVD decompositions: U, S, Vt, UR
        U: unitary matrix
        S: diagonal matrix
        Vt: Orthogonal matrix
        UR: U*Y

    alpha : {float, array-like},
        shape = [n_targets] if array-like
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number.

    sample_weight : float or numpy array of shape [n_samples]
        Individual weights for each sample. If sample_weight is not None and
        solver='auto', the solver will be set to 'cholesky'.

        .. versionadded:: 0.17


    Returns
    -------
    coef : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).

    Notes
    -----
    This function won't compute the intercept.
    """
    X, y, n_samples, n_features, n_targets, ravel = _check_data(X, y)
    if cache is None:
        u, s, vh, ur = _svd_and_cache(X, y)  # need to add error checking here
    else:
        u, s, vh, ur = cache
    has_sw = sample_weight is not None
    alpha = _check_alphas(alpha, n_targets)
    coef = _solve_cache_svd(vh, s, alpha, ur)
    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        coef = coef.ravel()
    return coef


def tikhonov_regression(X, y, alpha=1.0, sample_weight=None, L=None, method='analytic',
                     solver='auto', max_iter=None, tol=1e-3, verbose=0, random_state=None,
                     return_n_iter=False, return_intercept=False):
    # analytic solution (for reference, this will be inefficient)
    X, y, n_samples, n_features, n_targets, ravel = _check_data(X, y, solver)
    alpha = _check_alphas(alpha, n_targets)

    if L is None:
        L = np.diag(np.ones(n_features))

    if method is 'analytic':
        coef = np.empty([n_features, n_targets])
        for i in range(n_targets):
            coef[:, i] = reduce(np.dot,
                           [np.linalg.inv(np.dot(X.T, X) + alpha[i] * np.dot(L.T, L)),
                           X.T, y[:, i]])
    else:
        x_new, y_new, hq, kp, rp, ko, ho, to = to_standard_form(X, y, L)
        # choose ridge regression based on method
        b = ridge_regression(x_new, y_new, alpha, sample_weight=sample_weight, solver='auto',
                                        max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state,
                                        return_n_iter=return_n_iter, return_intercept=return_intercept)
        coef = to_general_form(X, y, b.T, kp, rp, ko, to, ho)

    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        coef = coef.ravel()
    return coef.T


class _BaseTikhonovReg(six.with_metaclass(ABCMeta, LinearModel)):

    @abstractmethod
    def __init__(self, gamma=None, alpha=1.0, fit_intercept=False, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, singcutoff=None,
                 solver='cache', cache=None,
                 random_state=None):
        self.gamma = gamma
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state
        self.cache = cache
        self.singcutoff = singcutoff

    def fit(self, X, y, alpha=None, sample_weight=None, singcutoff=None):
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64,
                         multi_output=True, y_numeric=True)

        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if alpha is not None:
            self.alpha = alpha

        if self.gamma is not None:
            X_orig, y_orig = X, y
            X, y, hq, kp, rp, ko, ho, to = to_standard_form(X, y, self.gamma)

        if self.solver is 'cache':
            self.cache = _svd_and_cache(X, y, singcutoff)
            self.coef_ = cache_ridge_regression(X, y, self.cache, self.alpha, sample_weight=sample_weight)
            self._set_intercept(X_offset, y_offset, X_scale)
            # temporary fix for fitting the intercept with sparse data using 'sag'
        else:
            if sparse.issparse(X) and self.fit_intercept:
                self.coef_, self.n_iter_, self.intercept_ = ridge_regression(
                    X, y, alpha=self.alpha, sample_weight=sample_weight,
                    max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                    random_state=self.random_state, return_n_iter=True,
                    return_intercept=True)
                self.intercept_ += y_offset
            else:
                self.coef_, self.n_iter_ = ridge_regression(
                    X, y, alpha=self.alpha, sample_weight=sample_weight,
                    max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                    random_state=self.random_state, return_n_iter=True,
                    return_intercept=False)
                self._set_intercept(X_offset, y_offset, X_scale)

        if self.gamma is not None:
            self.coef_ = to_general_form(X_orig, y_orig, self.coef_.T, kp, rp, ko, to, ho).T

        else:
            self.coef_ = self.coef_

        return self


class TikhonovReg(_BaseTikhonovReg, RegressorMixin):
    """Linear least squares with l2 regularization.

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape [n_samples, n_targets]).

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    gamma : {float, array-like}, shape (n_features, params)
        Regularization matrix; must be a positive float. This matrix
        is also known as the Tikhonov matrix
        Provides better conditioned regularization taking intrinsic covariance
        into account.

    alpha : {float, array-like}, shape (n_targets)
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.

    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'}
        Solver to use in the computational routines: (currently only SVD is implemented)

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than
          'cholesky'.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest but may not be available
          in old scipy versions. It also uses an iterative procedure.

        - 'sag' uses a Stochastic Average Gradient descent. It also uses an
          iterative procedure, and is often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' fast
          convergence is only guaranteed on features with approximately the
          same scale. You can preprocess the data with a scaler from
          sklearn.preprocessing.

        All last four solvers support both dense and sparse data. However,
        only 'sag' supports sparse input when `fit_intercept` is True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.

    tol : float
        Precision of the solution.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data. Used only in 'sag' solver.

        .. versionadded:: 0.17
           *random_state* to support Stochastic Average Gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    n_iter_ : array or None, shape (n_targets,)
        Actual number of iterations for each target. Available only for
        sag and lsqr solvers. Other solvers will return None.

        .. versionadded:: 0.17

    See also
    --------
    RidgeClassifier, RidgeCV, :class:`sklearn.kernel_ridge.KernelRidge`

    Examples
    --------
    >>> from sklearn.linear_model import TikhonovReg
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = TikhonovReg(alpha=1.0)
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)

    """

    def __init__(self, gamma=None, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver='cache',
                 random_state=None, singcutoff=None):
        super(TikhonovReg, self).__init__(gamma=gamma, alpha=alpha, fit_intercept=fit_intercept,
                                       normalize=normalize, copy_X=copy_X,
                                       max_iter=max_iter, tol=tol, solver=solver,
                                       random_state=random_state, singcutoff=singcutoff)

    def fit(self, X, y, sample_weight=None):
        """Fit Tikhonov regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or numpy array of shape [n_samples]
            Individual weights for each sample

        Returns
        -------
        self : returns an instance of self.
        """
        return super(TikhonovReg, self).fit(X, y, sample_weight=sample_weight)


class _TikhonovRegGCV(LinearModel):
    """Tikhonov regression with built-in Generalized Cross-Validation

    It allows efficient Leave-One-Out cross-validation.

    This class is not intended to be used directly. Use TikhonovCV instead.

    This implementation is identical to RidgeCV in scikit learn, but with rotation to standard/general space

    Notes
    -----

    We want to solve (K + alpha*Id)c = y,
    where K = X X^T is the kernel matrix.

    Let G = (K + alpha*Id)^-1.

    Dual solution: c = Gy
    Primal solution: w = X^T c

    Compute eigendecomposition K = Q V Q^T.
    Then G = Q (V + alpha*Id)^-1 Q^T,
    where (V + alpha*Id) is diagonal.
    It is thus inexpensive to inverse for many alphas.

    Let loov be the vector of prediction values for each example
    when the model was fitted with all examples but this example.

    loov = (KGY - diag(KG)Y) / diag(I-KG)

    Let looe be the vector of prediction errors for each example
    when the model was fitted with all examples but this example.

    looe = y - loov = c / diag(G)

    References
    ----------
    http://cbcl.mit.edu/projects/cbcl/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    http://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0),
                 fit_intercept=True, normalize=False,
                 scoring=None, copy_X=True,
                 gcv_mode=None, store_cv_values=False, gamma=None):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.gamma = gamma

    def _pre_compute(self, X, y):
        # even if X is very sparse, K is usually very dense
        K = safe_sparse_dot(X, X.T, dense_output=True)
        v, Q = linalg.eigh(K)
        QT_y = np.dot(Q.T, y)
        return v, Q, QT_y

    def _decomp_diag(self, v_prime, Q):
        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        return (v_prime * Q ** 2).sum(axis=-1)

    def _diag_dot(self, D, B):
        # compute dot(diag(D), B)
        if len(B.shape) > 1:
            # handle case where B is > 1-d
            D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
        return D * B

    def _errors_and_values_helper(self, alpha, y, v, Q, QT_y):
        """Helper function to avoid code duplication between self._errors and
        self._values.

        Notes
        -----
        We don't construct matrix G, instead compute action on y & diagonal.
        """
        w = 1.0 / (v + alpha)
        c = np.dot(Q, self._diag_dot(w, QT_y))
        G_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return (c / G_diag) ** 2, c

    def _values(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return y - (c / G_diag), c

    def _pre_compute_svd(self, X, y):
        if sparse.issparse(X):
            raise TypeError("SVD not supported for sparse matrices")
        U, s, _ = linalg.svd(X, full_matrices=0)
        v = s ** 2
        UT_y = np.dot(U.T, y)
        return v, U, UT_y

    def _errors_and_values_svd_helper(self, alpha, y, v, U, UT_y):
        """Helper function to avoid code duplication between self._errors_svd
        and self._values_svd.
        """
        w = ((v + alpha) ** -1) - (alpha ** -1)
        c = np.dot(U, self._diag_dot(w, UT_y)) + (alpha ** -1) * y
        G_diag = self._decomp_diag(w, U) + (alpha ** -1)
        if len(y.shape) != 1:
            # handle case where y is 2-d
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def _errors_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return (c / G_diag) ** 2, c

    def _values_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return y - (c / G_diag), c

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : Returns self.
        """
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64,
                         multi_output=True, y_numeric=True)

        n_samples, n_features = X.shape

        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        # probably there is no need to transform it back:
        if self.gamma is not None:
            X_orig, y_orig = X, y
            X, y, hq, kp, rp, ko, ho, to = to_standard_form(X, y, self.gamma)

        gcv_mode = self.gcv_mode
        with_sw = len(np.shape(sample_weight))

        if gcv_mode is None or gcv_mode == 'auto':
            if sparse.issparse(X) or n_features > n_samples or with_sw:
                gcv_mode = 'eigen'
            else:
                gcv_mode = 'svd'
        elif gcv_mode == "svd" and with_sw:
            # FIXME non-uniform sample weights not yet supported
            warnings.warn("non-uniform sample weights unsupported for svd, "
                          "forcing usage of eigen")
            gcv_mode = 'eigen'

        if gcv_mode == 'eigen':
            _pre_compute = self._pre_compute
            _errors = self._errors
            _values = self._values
        elif gcv_mode == 'svd':
            # assert n_samples >= n_features
            _pre_compute = self._pre_compute_svd
            _errors = self._errors_svd
            _values = self._values_svd
        else:
            raise ValueError('bad gcv_mode "%s"' % gcv_mode)

        v, Q, QT_y = _pre_compute(X, y)
        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        cv_values = np.zeros((n_samples * n_y, len(self.alphas)))
        C = []

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None

        for i, alpha in enumerate(self.alphas):
            weighted_alpha = (sample_weight * alpha
                              if sample_weight is not None
                              else alpha)
            if error:
                out, c = _errors(weighted_alpha, y, v, Q, QT_y)
            else:
                out, c = _values(weighted_alpha, y, v, Q, QT_y)
            cv_values[:, i] = out.ravel()
            C.append(c)

        if error:
            best = cv_values.mean(axis=0).argmin()
        else:
            # The scorer want an object that will make the predictions but
            # they are already computed efficiently by _RidgeGCV. This
            # identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.decision_function = lambda y_predict: y_predict
            identity_estimator.predict = lambda y_predict: y_predict

            out = [scorer(identity_estimator, y.ravel(), cv_values[:, i])
                   for i in range(len(self.alphas))]
            best = np.argmax(out)

        self.alpha_ = self.alphas[best]
        self.dual_coef_ = C[best]
        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)

        if self.gamma is not None:
            self.coef_ = to_general_form(X_orig, y_orig, self.coef_.T, kp, rp, ko, to, ho).T

        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = n_samples, len(self.alphas)
            else:
                cv_values_shape = n_samples, n_y, len(self.alphas)
            self.cv_values_ = cv_values.reshape(cv_values_shape)

        return self


class _BaseTikhonovRegCV(LinearModel):
    def __init__(self, alphas=(0.1, 1.0, 10.0), gamma=None,
                 fit_intercept=True, normalize=False, scoring=None,
                 cv=None, gcv_mode=None,
                 store_cv_values=False):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.gamma = gamma

    def fit(self, X, y, sample_weight=None):
        """Fit Tikhonov regression model

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : Returns self.
        """
        if self.cv is None:
            estimator = _TikhonovRegGCV(self.alphas,
                                  fit_intercept=self.fit_intercept,
                                  normalize=self.normalize,
                                  scoring=self.scoring,
                                  gcv_mode=self.gcv_mode,
                                  store_cv_values=self.store_cv_values,
                                  gamma = self.gamma)
            estimator.fit(X, y, sample_weight=sample_weight)
            self.alpha_ = estimator.alpha_
            if self.store_cv_values:
                self.cv_values_ = estimator.cv_values_
        else:
            if self.store_cv_values:
                raise ValueError("cv!=None and store_cv_values=True "
                                 " are incompatible")
            parameters = {'alpha': self.alphas}
            fit_params = {'sample_weight': sample_weight}
            gs = GridSearchCV(TikhonovReg(fit_intercept=self.fit_intercept),
                              parameters, fit_params=fit_params, cv=self.cv)
            gs.fit(X, y)
            estimator = gs.best_estimator_
            self.alpha_ = gs.best_estimator_.alpha

        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_

        return self


class TikhonovRegCV(_BaseTikhonovRegCV, RegressorMixin):
    """Tikhonov regression with built-in cross-validation.

    By default, it performs Generalized Cross-Validation, which is a form of
    efficient Leave-One-Out cross-validation.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alphas : numpy array of shape [n_alphas]
        Array of alpha values to try.
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC.

    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    gcv_mode : {None, 'auto', 'svd', eigen'}, optional
        Flag indicating which strategy to use when performing
        Generalized Cross-Validation. Options are::

            'auto' : use svd if n_samples > n_features or when X is a sparse
                     matrix, otherwise use eigen
            'svd' : force computation via singular value decomposition of X
                    (does not work for sparse matrices)
            'eigen' : force computation via eigendecomposition of X^T X

        The 'auto' mode is the default and is intended to pick the cheaper
        option of the two depending upon the shape and format of the training
        data.

    store_cv_values : boolean, default=False
        Flag indicating if the cross-validation values corresponding to
        each alpha should be stored in the `cv_values_` attribute (see
        below). This flag is only compatible with `cv=None` (i.e. using
        Generalized Cross-Validation).

    Attributes
    ----------
    cv_values_ : array, shape = [n_samples, n_alphas] or \
        shape = [n_samples, n_targets, n_alphas], optional
        Cross-validation values for each alpha (if `store_cv_values=True` and \
        `cv=None`). After `fit()` has been called, this attribute will \
        contain the mean squared errors (by default) or the values of the \
        `{loss,score}_func` function (if provided in the constructor).

    coef_ : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).

    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    alpha_ : float
        Estimated regularization parameter.

    See also
    --------
    Ridge: Ridge regression
    RidgeClassifier: Ridge classifier
    RidgeClassifierCV: Ridge classifier with built-in cross validation
    """
    pass



# def _ridgecv(X, y, alphas, grp=None, k=5, singcutoff=None):
#     # iterates through alphas and returns the y that has best prediction accuracy.
#     scores = []
#
#     if grp is not None:
#         from sklearn.model_selection import GroupKFold
#         gkf = GroupKFold(n_splits = k)
#         for train, test in gkf.split(X, y, groups=grp):
#             u, s, vh, ur = _svd_and_cache(X[train], y[train], singcutoff)
#             pvh = np.dot(y[test], vh.T)
#             for a in alphas:
#                 D = s / (s ** 2 + a ** 2)
#                 pred = reduce(np.dot, [pvh, np.diag(D), ur])
#                 rcorr = np.multiply(y, pred).mean(0)
#                 rcorr[np.isnan(rcorr)] = 0
#                 scores.append(rcorr)
#     else:
#         from sklearn.model_selection import KFold
#         kf = KFold(n_splits=k)
#         for train, test in kf.split(X):
#             u, s, vh, ur = _svd_and_cache(X[train], y[train], singcutoff)
#             pvh = np.dot(y[test], vh.T)
#             for a in alphas:
#                 D = s / (s**2 + a**2)
#                 pred = reduce(np.dot, [pvh, np.diag(D), ur])
#                 rcorr = np.multiply(y, pred).mean(0)
#                 rcorr[np.isnan(rcorr)] = 0
#                 scores.append(rcorr)
#     return scores
# def normal_ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
#                      max_iter=None, tol=1e-3, verbose=0, random_state=None,
#                      return_n_iter=False, return_intercept=False):
#     """Solve the ridge equation by the method of normal equations. Identical to the Ridge regression of scikit-learn
#
#     Read more in the :ref:`User Guide <ridge_regression>`.
#
#     Parameters
#     ----------
#     X : {array-like, sparse matrix, LinearOperator},
#         shape = [n_samples, n_features]
#         Training data
#
#     y : array-like, shape = [n_samples] or [n_samples, n_targets]
#         Target values
#
#     alpha : {float, array-like},
#         shape = [n_targets] if array-like
#         Regularization strength; must be a positive float. Regularization
#         improves the conditioning of the problem and reduces the variance of
#         the estimates. Larger values specify stronger regularization.
#         Alpha corresponds to ``C^-1`` in other linear models such as
#         LogisticRegression or LinearSVC. If an array is passed, penalties are
#         assumed to be specific to the targets. Hence they must correspond in
#         number.
#
#     max_iter : int, optional
#         Maximum number of iterations for conjugate gradient solver.
#         For 'sparse_cg' and 'lsqr' solvers, the default value is determined
#         by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
#
#     sample_weight : float or numpy array of shape [n_samples]
#         Individual weights for each sample. If sample_weight is not None and
#         solver='auto', the solver will be set to 'cholesky'.
#
#         .. versionadded:: 0.17
#
#     solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'}
#         Solver to use in the computational routines:
#
#         - 'auto' chooses the solver automatically based on the type of data.
#
#         - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
#           coefficients. More stable for singular matrices than
#           'cholesky'.
#
#         - 'cholesky' uses the standard scipy.linalg.solve function to
#           obtain a closed-form solution via a Cholesky decomposition of
#           dot(X.T, X)
#
#         - 'sparse_cg' uses the conjugate gradient solver as found in
#           scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
#           more appropriate than 'cholesky' for large-scale data
#           (possibility to set `tol` and `max_iter`).
#
#         - 'lsqr' uses the dedicated regularized least-squares routine
#           scipy.sparse.linalg.lsqr. It is the fastest but may not be available
#           in old scipy versions. It also uses an iterative procedure.
#
#         - 'sag' uses a Stochastic Average Gradient descent. It also uses an
#           iterative procedure, and is often faster than other solvers when
#           both n_samples and n_features are large. Note that 'sag' fast
#           convergence is only guaranteed on features with approximately the
#           same scale. You can preprocess the data with a scaler from
#           sklearn.preprocessing.
#
#         All last four solvers support both dense and sparse data. However,
#         only 'sag' supports sparse input when `fit_intercept` is True.
#
#         .. versionadded:: 0.17
#            Stochastic Average Gradient descent solver.
#
#     tol : float
#         Precision of the solution.
#
#     verbose : int
#         Verbosity level. Setting verbose > 0 will display additional
#         information depending on the solver used.
#
#     random_state : int seed, RandomState instance, or None (default)
#         The seed of the pseudo random number generator to use when
#         shuffling the data. Used only in 'sag' solver.
#
#     return_n_iter : boolean, default False
#         If True, the method also returns `n_iter`, the actual number of
#         iteration performed by the solver.
#
#         .. versionadded:: 0.17
#
#     return_intercept : boolean, default False
#         If True and if X is sparse, the method also returns the intercept,
#         and the solver is automatically changed to 'sag'. This is only a
#         temporary fix for fitting the intercept with sparse data. For dense
#         data, use sklearn.linear_model._preprocess_data before your regression.
#
#         .. versionadded:: 0.17
#
#     Returns
#     -------
#     coef : array, shape = [n_features] or [n_targets, n_features]
#         Weight vector(s).
#
#     n_iter : int, optional
#         The actual number of iteration performed by the solver.
#         Only returned if `return_n_iter` is True.
#
#     intercept : float or array, shape = [n_targets]
#         The intercept of the model. Only returned if `return_intercept`
#         is True and if X is a scipy sparse array.
#
#     Notes
#     -----
#     This function won't compute the intercept.
#     """
#
#     X, y, n_samples, n_features, n_targets, ravel = _check_data(X, y, solver)
#     has_sw = sample_weight is not None
#
#     if solver == 'auto':
#         # cholesky if it's a dense array and cg in any other case
#         if not sparse.issparse(X) or has_sw:
#             solver = 'cholesky'
#         else:
#             solver = 'sparse_cg'
#
#     elif solver == 'lsqr' and not hasattr(sp_linalg, 'lsqr'):
#         warnings.warn("""lsqr not available on this machine, falling back
#                       to sparse_cg.""")
#         solver = 'sparse_cg'
#
#     if has_sw:
#         if np.atleast_1d(sample_weight).ndim > 1:
#             raise ValueError("Sample weights must be 1D array or scalar")
#
#         if solver != 'sag':
#             # SAG supports sample_weight directly. For other solvers,
#             # we implement sample_weight via a simple rescaling.
#             X, y = _rescale_data(X, y, sample_weight)
#
#     # There should be either 1 or n_targets penalties
#     alpha = np.asarray(alpha).ravel()
#     if alpha.size not in [1, n_targets]:
#         raise ValueError("Number of targets and number of penalties "
#                          "do not correspond: %d != %d"
#                          % (alpha.size, n_targets))
#
#     if alpha.size == 1 and n_targets > 1:
#         alpha = np.repeat(alpha, n_targets)
#
#     if solver not in ('sparse_cg', 'cholesky', 'svd', 'lsqr', 'sag'):
#         raise ValueError('Solver %s not understood' % solver)
#
#     n_iter = None
#     if solver == 'sparse_cg':
#         coef = _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose)
#
#     elif solver == 'lsqr':
#         coef, n_iter = _solve_lsqr(X, y, alpha, max_iter, tol)
#
#     elif solver == 'cholesky':
#         if n_features > n_samples:
#             K = safe_sparse_dot(X, X.T, dense_output=True)
#             try:
#                 dual_coef = _solve_cholesky_kernel(K, y, alpha)
#
#                 coef = safe_sparse_dot(X.T, dual_coef, dense_output=True).T
#             except linalg.LinAlgError:
#                 # use SVD solver if matrix is singular
#                 solver = 'svd'
#
#         else:
#             try:
#                 coef = _solve_cholesky(X, y, alpha)
#             except linalg.LinAlgError:
#                 # use SVD solver if matrix is singular
#                 solver = 'svd'
#
#     elif solver == 'sag':
#         # precompute max_squared_sum for all targets
#         max_squared_sum = row_norms(X, squared=True).max()
#
#         coef = np.empty((y.shape[1], n_features))
#         n_iter = np.empty(y.shape[1], dtype=np.int32)
#         intercept = np.zeros((y.shape[1], ))
#         for i, (alpha_i, target) in enumerate(zip(alpha, y.T)):
#             init = {'coef': np.zeros((n_features + int(return_intercept), 1))}
#             coef_, n_iter_, _ = sag_solver(
#                 X, target.ravel(), sample_weight, 'squared', alpha_i,
#                 max_iter, tol, verbose, random_state, False, max_squared_sum,
#                 init)
#             if return_intercept:
#                 coef[i] = coef_[:-1]
#                 intercept[i] = coef_[-1]
#             else:
#                 coef[i] = coef_
#             n_iter[i] = n_iter_
#
#         if intercept.shape[0] == 1:
#             intercept = intercept[0]
#         coef = np.asarray(coef)
#
#     if solver == 'svd':
#         if sparse.issparse(X):
#             raise TypeError('SVD solver does not support sparse'
#                             ' inputs currently')
#         coef = _solve_svd(X, y, alpha)
#
#     if ravel:
#         # When y was passed as a 1d-array, we flatten the coefficients.
#         coef = coef.ravel()
#
#     if return_n_iter and return_intercept:
#         return coef, n_iter, intercept
#     elif return_intercept:
#         return coef, intercept
#     elif return_n_iter:
#         return coef, n_iter
#     else:
#         return coef
# working here