#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 4
##################################################################

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

rates = [1, 2, 5]
scales = [1, 2, 3]

x = np.linspace(0, 20, 100)
f, ax = plt.subplots(len(rates), len(scales), sharex=True, sharey=True)
for i in range(len(rates)):
  for j in range(len(scales)):
    rate = rates[i]
    scale = scales[j]
    rv = stats.gamma(a=rate, scale=scale)
    ax[i,j].plot(x, rv.pdf(x))
    ax[i,j].plot(0, 0, label="$\\alpha$ = {:3.2f}\n$\\theta$ = {:3.2f}".format(rate, scale), alpha=0)
    ax[i,j].legend()

ax[2,1].set_xlabel('$x$')
ax[1,0].set_ylabel('$pdf(x)$')
plt.savefig('img401.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b.')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.plot(x, y_real, 'k')
plt.subplot(1, 2, 2)
sns.kdeplot(y)
plt.xlabel('$y$', fontsize=16)
plt.savefig('img403.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = pm.Deterministic('mu', alpha + beta * x)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

  start = pm.find_MAP()
  step = pm.Metropolis()
  trace = pm.sample(11000, step, start, njobs=1)

trace_n = trace[1000:]
pm.traceplot(trace_n)
plt.savefig('img404.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

varnames = ['alpha', 'beta', 'epsilon']
pm.autocorrplot(trace_n, varnames)
plt.savefig('img405.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

sns.kdeplot(trace_n['alpha'], trace_n['beta'])
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\beta$', fontsize=16, rotation=0)
plt.savefig('img406.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

plt.plot(x, y, 'b.');
alpha_m = trace_n['alpha'].mean()
beta_m = trace_n['beta'].mean()
plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=2, fontsize=14)
plt.savefig('img407.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

plt.plot(x, y, 'b.');
idx = range(0, len(trace_n['alpha']), 10)
plt.plot(x, trace_n['alpha'][idx] + trace_n['beta'][idx] * x[:,np.newaxis], c='gray', alpha=0.5);

plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=2, fontsize=14)
plt.savefig('img408.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))

idx = np.argsort(x)
x_ord = x[idx]
sig = pm.hpd(trace_n['mu'], alpha=.02)[idx]
plt.fill_between(x_ord, sig[:,0], sig[:,1], color='gray')

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img409.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

ppc = pm.sample_ppc(trace_n, samples=1000, model=model)
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k', label='y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))

sig0 = pm.hpd(ppc['y_pred'], alpha=0.5)[idx]
sig1 = pm.hpd(ppc['y_pred'], alpha=0.05)[idx]
plt.fill_between(x_ord, sig0[:,0], sig0[:,1], color='gray', alpha=1)
plt.fill_between(x_ord, sig1[:,0], sig1[:,1], color='gray', alpha=0.5)

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img410.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

with pm.Model() as model_n:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = alpha + beta * x
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

  rb = pm.Deterministic('rb', (beta * x.std() / y.std()) ** 2)

  y_mean = y.mean()
  ss_reg = pm.math.sum((mu - y_mean) ** 2)
  ss_tot = pm.math.sum((y - y_mean) ** 2)
  rss = pm.Deterministic('rss', ss_reg/ss_tot)

  start = pm.find_MAP()
  step = pm.NUTS()
  trace_n = pm.sample(2000, step=step, start=start)

pm.traceplot(trace_n)
plt.savefig('img411.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

varnames = ['alpha', 'beta', 'epsilon', 'rb', 'rss']
pm.summary(trace_n, varnames)

sigma_x1 = 1
sigmas_x2 = [1, 2]
rhos = [-0.99, -0.5, 0, 0.5, 0.99]

x, y = np.mgrid[-5:5:.1, -5:5:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

f, ax = plt.subplots(len(sigmas_x2), len(rhos), sharex=True, sharey=True)

for i in range(2):
  for j in range(5):
    sigma_x2 = sigmas_x2[i]
    rho = rhos[j]
    cov = [[sigma_x1**2, sigma_x1*sigma_x2*rho], [sigma_x1*sigma_x2*rho, sigma_x2**2]]
    rv = stats.multivariate_normal([0, 0], cov)
    ax[i,j].contour(x, y, rv.pdf(pos))
    ax[i,j].plot(0, 0, label="$\\sigma_{{x2}}$ = {:3.2f}\n$\\rho$ = {:3.2f}".format(sigma_x2, rho), alpha=0)
    ax[i,j].legend()
ax[1,2].set_xlabel('$x_1$')
ax[1,0].set_ylabel('$x_2$')
plt.savefig('img412.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x 
y = y_real + eps_real

data = np.stack((x, y)).T

with pm.Model() as pearson_model:
  mu = pm.Normal('mu', mu=data.mean(0), sd=10, shape=2)
  sigma_1 = pm.HalfNormal('simga_1', 10)
  sigma_2 = pm.HalfNormal('sigma_2', 10)
  rho = pm.Uniform('rho', -1, 1)
  
  cov = pm.math.stack(([sigma_1**2, sigma_1*sigma_2*rho], [sigma_1*sigma_2*rho, sigma_2**2]))
  
  y_pred = pm.MvNormal('y_pred', mu=mu, cov=cov, observed=data)
  
  start = pm.find_MAP()
  step = pm.NUTS(scaling=start)
  trace_p = pm.sample(1000, step=step, start=start, njobs=1)

pm.traceplot(trace_p)
plt.savefig('img413.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

ans = sns.load_dataset('anscombe')
x_3 = ans[ans.dataset == 'III']['x'].values
y_3 = ans[ans.dataset == 'III']['y'].values
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
plt.plot(x_3, (alpha_c + beta_c* x_3), 'k', label='y ={:.2f} + {:.2f} * x'.format(alpha_c, beta_c))
plt.plot(x_3, y_3, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', rotation=0, fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.subplot(1,2,2)
sns.kdeplot(y_3);
plt.xlabel('$y$', fontsize=16)
plt.tight_layout()
plt.savefig('img414.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_t:
  alpha = pm.Normal('alpha', mu=0, sd=100)
  beta = pm.Normal('beta', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)
  nu = pm.Deterministic('nu', pm.Exponential('nu_', 1/29) + 1)
  
  y_pred = pm.StudentT('y_pred', mu=alpha + beta * x_3, sd=epsilon, nu=nu, observed=y_3)
  
  start = pm.find_MAP()
  step = pm.NUTS(scaling=start) 
  trace_t = pm.sample(2000, step=step, start=start, njobs=1)

beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]

plt.plot(x_3, (alpha_c + beta_c * x_3), 'k', label='non-robust', alpha=0.5)
plt.plot(x_3, y_3, 'bo')
alpha_m = trace_t['alpha'].mean()
beta_m = trace_t['beta'].mean()
plt.plot(x_3, alpha_m + beta_m * x_3, c='k', label='robust')

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', rotation=0, fontsize=16)
plt.legend(loc=2, fontsize=12)
plt.savefig('img415.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

ppc = pm.sample_ppc(trace_t, samples=200, model=model_t, random_seed=2)
for y_tilde in ppc['y_pred']:
  sns.kdeplot(y_tilde, alpha=0.5, c='g')

sns.kdeplot(y_3, linewidth=3)
plt.savefig('img416.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

N = 20
M = 8
idx = np.repeat(range(M-1), N)
idx = np.append(idx, 7)
np.random.seed(314)

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m  + eps_real

plt.figure(figsize=(16,8))
j, k = 0, N
for i in range(M):
  plt.subplot(2,4,i+1)
  plt.scatter(x_m[j:k], y_m[j:k])
  plt.xlabel('$x_{}$'.format(i), fontsize=16)
  plt.ylabel('$y$', fontsize=16, rotation=0)
  plt.xlim(6, 15)
  plt.ylim(7, 17)
  j += N
  k += N
plt.tight_layout()
plt.savefig('img417.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

x_centered = x_m - x_m.mean()

with pm.Model() as unpooled_model:
  alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=10, shape=M)
  beta = pm.Normal('beta', mu=0, sd=10, shape=M)
  epsilon = pm.HalfCauchy('epsilon', 5)
  nu = pm.Exponential('nu', 1/30)
  y_pred = pm.StudentT('y_pred', mu= alpha_tmp[idx] + beta[idx] * x_centered, sd=epsilon, nu=nu, observed=y_m)
  alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_m.mean()) 
    
  start = pm.find_MAP()
  step = pm.NUTS(scaling=start)
  trace_up = pm.sample(2000, step=step, start=start, njobs=1)

varnames=['alpha', 'beta', 'epsilon', 'nu']
pm.traceplot(trace_up, varnames)
plt.savefig('img418.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as hierarchical_model:
  alpha_tmp_mu = pm.Normal('alpha_tmp_mu', mu=0, sd=10)
  alpha_tmp_sd = pm.HalfNormal('alpha_tmp_sd', 10)
  beta_mu = pm.Normal('beta_mu', mu=0, sd=10)
  beta_sd = pm.HalfNormal('beta_sd', sd=10)

  alpha_tmp = pm.Normal('alpha_tmp', mu=alpha_tmp_mu, sd=alpha_tmp_sd, shape=M)
  beta = pm.Normal('beta', mu=beta_mu, sd=beta_sd, shape=M)
  epsilon = pm.HalfCauchy('epsilon', 5)
  nu = pm.Exponential('nu', 1/30)

  y_pred = pm.StudentT('y_pred', mu=alpha_tmp[idx] + beta[idx] * x_centered, sd=epsilon, nu=nu, observed=y_m)

  alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_m.mean()) 
  alpha_mu = pm.Deterministic('alpha_mu', alpha_tmp_mu - beta_mu * x_m.mean())
  alpha_sd = pm.Deterministic('alpha_sd', alpha_tmp_sd - beta_mu * x_m.mean())
  
  trace_hm = pm.sample(1000, njobs=1)

varnames=['alpha', 'alpha_mu', 'alpha_sd', 'beta', 'beta_mu', 'beta_sd', 'epsilon', 'nu']
pm.traceplot(trace_hm, varnames)
plt.savefig('img420.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

plt.figure(figsize=(16,8))
j, k = 0, N
x_range = np.linspace(x_m.min(), x_m.max(), 10)
for i in range(M):
  plt.subplot(2,4,i+1)
  plt.scatter(x_m[j:k], y_m[j:k])
  plt.xlabel('$x_{}$'.format(i), fontsize=16)
  plt.ylabel('$y$', fontsize=16, rotation=0)
  alfa_m = trace_hm['alpha'][:,i].mean()
  beta_m = trace_hm['beta'][:,i].mean()
  plt.plot(x_range, alfa_m + beta_m * x_range, c='k', label='y = {:.2f} + {:.2f} * x'.format(alfa_m, beta_m))
  plt.xlim(x_m.min()-1, x_m.max()+1)
  plt.ylim(y_m.min()-1, y_m.max()+1)
  j += N
  k += N
plt.tight_layout()
plt.savefig('img421.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

ans = sns.load_dataset('anscombe')
x_2 = ans[ans.dataset == 'II']['x'].values
y_2 = ans[ans.dataset == 'II']['y'].values
x_2 = x_2 - x_2.mean()
y_2 = y_2 - y_2.mean()

plt.scatter(x_2, y_2)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img422.png')

plt.figure()

with pm.Model() as model_poly:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta1 = pm.Normal('beta1', mu=0, sd=1)
  beta2 = pm.Normal('beta2', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = alpha + beta1 * x_2 + beta2 * x_2**2
  
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_2)

  trace_poly = pm.sample(2000, njobs=1)

pm.traceplot(trace_poly)
plt.tight_layout()
plt.savefig('img423.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

x_p = np.linspace(-6, 6)
y_p = trace_poly['alpha'].mean() + trace_poly['beta1'].mean() * x_p + trace_poly['beta2'].mean() * x_p**2
plt.scatter(x_2, y_2)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.plot(x_p, y_p, c='r')
plt.savefig('img424.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

np.random.seed(314)
N = 100
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)

X = np.array([np.random.normal(i, j, N) for i,j in zip([10, 2], [1, 1.5])])
X_mean = X.mean(axis=1, keepdims=True)
X_centered = X - X_mean
y = alpha_real + np.dot(beta_real, X) + eps_real

def scatter_plot(x, y):
  plt.figure(figsize=(10, 10))
  for idx, x_i in enumerate(x):
    plt.subplot(2, 2, idx+1)
    plt.scatter(x_i, y)
    plt.xlabel('$x_{}$'.format(idx+1), fontsize=16)
    plt.ylabel('$y$', rotation=0, fontsize=16)

  plt.subplot(2, 2, idx+2)
  plt.scatter(x[0], x[1])
  plt.xlabel('$x_{}$'.format(idx), fontsize=16)
  plt.ylabel('$x_{}$'.format(idx+1), rotation=0, fontsize=16)

scatter_plot(X_centered, y)
plt.savefig('img425.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_mlr:
  alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=1, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)
  
  mu = alpha_tmp + pm.math.dot(beta, X_centered)
  alpha = pm.Deterministic('alpha', alpha_tmp - pm.math.dot(beta, X_mean)) 
  
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

  trace_mlr = pm.sample(5000, njobs=1)

varnames = ['alpha', 'beta','epsilon']
pm.traceplot(trace_mlr, varnames)
plt.savefig('img426.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.summary(trace_mlr, varnames)

np.random.seed(314)
N = 100
x_1 = np.random.normal(size=N)
x_2 = x_1 + np.random.normal(size=N, scale=1)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2))
scatter_plot(X, y)
plt.savefig('img427.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_red:
  alpha = pm.Normal('alpha', mu=0, sd=1)
  beta = pm.Normal('beta', mu=0, sd=10, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)
  
  mu = alpha + pm.math.dot(beta, X)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
  
  trace_red = pm.sample(5000, njobs=1)

pm.traceplot(trace_red)
plt.savefig('img428.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.summary(trace_red)

plt.figure()

x_2 = x_1 + np.random.normal(size=N, scale=0.01)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2))
scatter_plot(X, y)
plt.tight_layout()
plt.savefig('img427-1.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_red:
  alpha = pm.Normal('alpha', mu=0, sd=1)
  beta = pm.Normal('beta', mu=0, sd=10, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = alpha + pm.math.dot(beta, X)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

  trace_red = pm.sample(5000, njobs=1)

pm.traceplot(trace_red)
plt.tight_layout()
plt.savefig('img428-1.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

sns.kdeplot(trace_red['beta'][:,0], trace_red['beta'][:,1])
plt.xlabel(r'$\beta_1$', fontsize=16)
plt.ylabel(r'$\beta_2$', fontsize=16, rotation=0)
plt.savefig('img429.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.forestplot(trace_red, varnames=['beta'])
plt.savefig('img430.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

np.random.seed(314)
N = 100
r = 0.8

x_1 = np.random.normal(size=N)
x_2 = np.random.normal(loc=x_1 * r, scale=(1 - r ** 2) ** 0.5)
y = np.random.normal(loc=x_1 - x_2)
X = np.vstack((x_1, x_2))

scatter_plot(X, y)
plt.savefig('img431.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with pm.Model() as model_ma:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=10, shape=2)
  epsilon = pm.HalfCauchy('epsilon', 5)

  mu = alpha + pm.math.dot(beta, X)
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

  trace_ma = pm.sample(5000, njobs=1)

pm.traceplot(trace_ma)
plt.savefig('img432.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.forestplot(trace_ma, varnames=['beta']);
plt.savefig('img433.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

