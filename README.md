# tikhonov

L2-regularized regression using a non-diagonal regularization matrix

Linear least squares with l2 regularization.
This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. When the regularization matrix is a scalar multiple of the identity matrix, this is known as Ridge Regression. The general case, with an arbitrary regularization matrix (of full rank) is known as Tikhonov regularization. This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape [n_samples, n_targets]) and is based on the Ridge regression implementation of scikit-learn.

This specific implementation uses a transformation to standard for for efficiency, see the following refs:

Hansen, PC. (1994).Regularization tools: a Matlab package for analysis and solution of discrete ill-posed problems. Num. Algorithms; 6: 1-35

Stout, F., Kalivas, JH. (2006). Tikhonov regularization in standardized and general form for multivariate calibration with application towards removing unwanted spectral artifacts. Journal of Chemometrics; 20: 22-23.

Usage guide to follow.
