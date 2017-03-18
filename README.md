# tikhonov
code for L2 regularization of arbitrary Tikhonov matrices

Tikhonov regression

L2-regularized regression using a non-diagonal regularization matrix using (Truncated) SVD

Linear least squares with l2 regularization.
This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization. This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape [n_samples, n_targets]).

This specific implementation uses a transformation to standard for for efficiency, see the following ref:
Hansen, PC. (1994).Regularization tools: a Matlab package for analysis and solution of discrete ill-posed problems. Num. Algorithms; 6: 1-35

This function also includes an efficient cross-validation function and SVD caching functionality for fitting multivariate data with multiple regularization parameters.
