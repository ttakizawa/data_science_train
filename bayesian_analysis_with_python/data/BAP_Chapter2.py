#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 2
##################################################################

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

def posterior_grid(grid_points=100, heads=6, tosses=9):
  """
  A grid implementation for the coin-flip problem
  """
  grid = np.linspace(0, 1, grid_points)
  prior = np.repeat(5, grid_points)
  likelihood = stats.binom.pmf(heads, tosses, grid)
  unstd_posterior = likelihood * prior
  posterior = unstd_posterior / unstd_posterior.sum()
  return grid, posterior

points = 15
h, n = 1, 4
grid, posterior = posterior_grid(points, h, n)
plt.plot(grid, posterior, 'o-', label='heads = {}\ntosses = {}'.format(h, n))
plt.xlabel(r'$\theta$', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.savefig('img201.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

N = 10000
x, y = np.random.uniform(-1, 1, size=(2, N))
inside = (x**2 + y**2) <= 1
pi = inside.sum()*4/N
error = abs((pi - np.pi)/pi)* 100

outside = np.invert(inside)

plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.plot(0, 0, label='$\hat\pi$ = {:4.3f}\nerror = {:4.3f}%'.format(pi, error), alpha=0)
plt.axis('square')
plt.legend(frameon=True, framealpha=0.9, fontsize=16);
plt.savefig('img202.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

def metropolis(func, steps=10000):
  """A very simple Metropolis implementation"""
  samples = np.zeros(steps)
  old_x = func.mean()
  old_prob = func.pdf(old_x)

  for i in range(steps):
    new_x = old_x + np.random.normal(0, 0.5)
    new_prob = func.pdf(new_x)
    acceptance = new_prob/old_prob
    if acceptance >= np.random.random():
      samples[i] = new_x
      old_x = new_x
      old_prob = new_prob
    else:
      samples[i] = old_x
  return samples

func = stats.beta(0.4, 2)
samples = metropolis(func=func)
x = np.linspace(0.01, .99, 100)
y = func.pdf(x)
plt.xlim(0, 1)
plt.plot(x, y, 'r-', lw=3, label='True distribution')
plt.hist(samples, bins=30, normed=True, label='Estimated distribution')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$pdf(x)$', fontsize=14)
plt.legend(fontsize=14);
plt.savefig('img203.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

np.random.seed(123)
n_experiments = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
print(data)

with pm.Model() as our_first_model:
  theta = pm.Beta('theta', alpha=1, beta=1)
  y = pm.Bernoulli('y', p=theta, observed=data)

  start = pm.find_MAP()
  step = pm.Metropolis()
  trace = pm.sample(1000, step=step, start=start)

burnin = 100
chain = trace[burnin:]
pm.traceplot(chain, lines={'theta':theta_real});
plt.savefig('img204.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

with our_first_model:
  step = pm.Metropolis()
  multi_trace = pm.sample(1000, step=step, njobs=4)

burnin = 100
multi_chain = multi_trace[burnin:]
pm.traceplot(multi_chain, lines={'theta':theta_real});
plt.savefig('img206.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.gelman_rubin(multi_chain) 
{'theta': 1.0074579751170656, 'theta_logodds': 1.009770031607315}

pm.forestplot(multi_chain, varnames={'theta'});
plt.savefig('img207.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

#pm.df_summary(multi_chain)
pm.summary(multi_chain)

pm.autocorrplot(chain)
plt.savefig('img208.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.effective_n(multi_chain)['theta']
pm.plot_posterior(chain, kde_plot=True)
plt.savefig('img209.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.plot_posterior(chain, kde_plot=True, rope=[0.45,.55])
plt.savefig('img210.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.plot_posterior(chain, kde_plot=True, ref_val=0.5)
plt.savefig('img211.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

