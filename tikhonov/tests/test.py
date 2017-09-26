import numpy as np
import TikhonovRegression as tik
from sklearn.linear_model import Ridge, LinearRegression

Sigma = np.array([[1, 0, .5], [0, 1, .5], [.5, .5, 1]])
n = 10
nf = Sigma.shape[0]
x = np.random.rand(n, nf)
b = np.random.multivariate_normal(np.zeros(nf), Sigma)

x = x - x.mean(0)
y = np.dot(x, b) + np.random.randn(n)/10

ols = LinearRegression(fit_intercept=False)
ridge = Ridge(fit_intercept=False)

o = ols.fit(x, y).coef_
t = ridge.fit(x, y).coef_
te = tik.analytic_tikhonov(x, y, 1, np.eye(nf))
ta = tik.analytic_tikhonov(x, y, 1, Sigma)

gamma = tik.find_gamma(Sigma)

x_new, y_new = tik.to_standard_form(x, y, gamma)
ta_est_standard = ridge.fit(x_new, y_new).coef_
ta_est = tik.to_general_form(ta_est_standard, x, y, gamma)

# verify that ta_est and ta are the sample
# verify that te and t are the sample
# verify that ta and te are close
