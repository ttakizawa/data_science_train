#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 8
##################################################################

import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
np.set_printoptions(precision=2)

np.random.seed(1)
x = np.random.uniform(0, 10, size=20)
y = np.random.normal(np.sin(x), 0.2)
plt.plot(x, y, 'o')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img801.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

def gauss_kernel(x, n_knots):
  """
  Simple Gaussian radial kernel
  """
  knots = np.linspace(x.min(), x.max(), n_knots)    
  w = 2 
  return np.array([np.exp(-(x-k)**2/w) for k in knots])

n_knots = 5

with pm.Model() as kernel_model:
  gamma = pm.Cauchy('gamma', alpha=0, beta=1, shape=n_knots)
  sd = pm.Uniform('sd',0,  10)
  mu = pm.math.dot(gamma, gauss_kernel(x, n_knots))
  yl = pm.Normal('yl', mu=mu, sd=sd, observed=y)

  kernel_trace = pm.sample(5000, njobs=1)

pm.traceplot(kernel_trace);
plt.savefig('img802.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

ppc = pm.sample_ppc(kernel_trace, model=kernel_model, samples=100)

plt.plot(x, ppc['yl'].T, 'ro', alpha=0.1)

plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img803.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

new_x = np.linspace(x.min(), x.max(), 100)
k = gauss_kernel(new_x, n_knots)
gamma_pred = kernel_trace['gamma']
for i in range(100):
  idx = np.random.randint(0, len(gamma_pred))
  y_pred = np.dot(gamma_pred[idx], k)
  plt.plot(new_x, y_pred, 'r-', alpha=0.1)
plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img804.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

squared_distance = lambda x, y: np.array([[(x[i] - y[j])**2 for i in range(len(x))] for j in range(len(y))])
np.random.seed(1)
test_points = np.linspace(0, 10, 100)
cov = np.exp(-squared_distance(test_points, test_points))
plt.plot(test_points, stats.multivariate_normal.rvs(cov=cov, size=6).T)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img805.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

np.random.seed(1)
eta = 1
rho = 0.5
sigma = 0.03
D = squared_distance(test_points, test_points)
cov = eta * np.exp(-rho * D)
diag = eta * sigma
np.fill_diagonal(cov, diag)

for i in range(6):
  plt.plot(test_points, stats.multivariate_normal.rvs(cov=cov))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img806.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

np.random.seed(1)
K_oo = eta * np.exp(-rho * D) 
D_x = squared_distance(x, x)
K = eta * np.exp(-rho * D_x)
diag_x = eta + sigma
np.fill_diagonal(K, diag_x)

D_off_diag = squared_distance(x, test_points)
K_o = eta * np.exp(-rho * D_off_diag)

mu_post = np.dot(np.dot(K_o, np.linalg.inv(K)), y)
SIGMA_post = K_oo - np.dot(np.dot(K_o, np.linalg.inv(K)), K_o.T)

for i in range(100):
  fx = stats.multivariate_normal.rvs(mean=mu_post, cov=SIGMA_post)
  plt.plot(test_points, fx, 'r-', alpha=0.1)

plt.plot(x, y, 'o')
 
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img807.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

np.random.seed(1)
eta = 1
rho = 0.5
sigma = 0.03

f = lambda x: np.sin(x).flatten()

def kernel(a, b):
  """ GP squared exponential kernel """
  kernelParameter = 0.1
  sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
  return eta * np.exp(- rho * sqdist)

N = 20
n = 100

X = np.random.uniform(0, 10, size=(N,1))
yfx = f(X) + sigma * np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + sigma * np.eye(N))

Xtest = np.linspace(0, 10, n).reshape(-1,1)

Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, yfx))

K_ = kernel(Xtest, Xtest)
sd_pred = (np.diag(K_) - np.sum(Lk**2, axis=0))**0.5

plt.fill_between(Xtest.flat, mu - 2 * sd_pred, mu + 2 * sd_pred, color="r", alpha=0.2)
plt.plot(Xtest, mu, 'r', lw=2)
plt.plot(x, y, 'o')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img808.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as GP:
  mu = np.zeros(N)
  eta = pm.HalfCauchy('eta', 5)
  rho = pm.HalfCauchy('rho', 5)
  sigma = pm.HalfCauchy('sigma', 5)
  
  D = squared_distance(x, x)
  K = tt.fill_diagonal(eta * pm.math.exp(-rho * D), eta + sigma)
  
  obs = pm.MvNormal('obs', mu, cov=K, observed=y)
  test_points = np.linspace(0, 10, 100)
  D_pred = squared_distance(test_points, test_points)
  D_off_diag = squared_distance(x, test_points)
  
  K_oo = eta * pm.math.exp(-rho * D_pred)
  K_o = eta * pm.math.exp(-rho * D_off_diag)
  
  mu_post = pm.Deterministic('mu_post', pm.math.dot(pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), y))
  SIGMA_post = pm.Deterministic('SIGMA_post', K_oo - pm.math.dot(pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), K_o.T))
  
  trace = pm.sample(1000, njobs=1)

varnames = ['eta', 'rho', 'sigma']
chain = trace[100:]
pm.traceplot(chain, varnames)
plt.savefig('img809.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

pm.summary(chain, varnames).round(4)

y_pred = [np.random.multivariate_normal(m, S) for m,S in zip(chain['mu_post'][::5], chain['SIGMA_post'][::5])]

for yp in y_pred:
  plt.plot(test_points, yp, 'r-', alpha=0.1)

plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img810.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

periodic = lambda x, y: np.array([[np.sin((x[i] - y[j])/2)**2 for i in range(len(x))] for j in range(len(y))])

with pm.Model() as GP_periodic:
  mu = np.zeros(N)
  eta = pm.HalfCauchy('eta', 5)
  rho = pm.HalfCauchy('rho', 5)
  sigma = pm.HalfCauchy('sigma', 5)
  
  P = periodic(x, x)
  K = tt.fill_diagonal(eta * pm.math.exp(-rho * P), eta + sigma)
  
  obs = pm.MvNormal('obs', mu, cov=K, observed=y)
  test_points = np.linspace(0, 10, 100)
  D_pred = periodic(test_points, test_points)
  D_off_diag = periodic(x, test_points)
  
  K_oo = eta * pm.math.exp(-rho * D_pred)
  K_o = eta * pm.math.exp(-rho * D_off_diag)
  
  mu_post = pm.Deterministic('mu_post', pm.math.dot(pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), y))
  SIGMA_post = pm.Deterministic('SIGMA_post', K_oo - pm.math.dot(pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), K_o.T))
  
  trace = pm.sample(1000, njobs=1)

varnames = ['eta', 'rho', 'sigma']
chain = trace[100:]
pm.traceplot(chain, varnames);
plt.savefig('img811-0.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

y_pred = [np.random.multivariate_normal(m, S) for m,S in zip(chain['mu_post'][::5], chain['SIGMA_post'][::5])]

for yp in y_pred:
    plt.plot(test_points, yp, 'r-', alpha=0.1)

plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img811.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

