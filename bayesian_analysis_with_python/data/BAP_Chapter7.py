#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 7
##################################################################

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
np.set_printoptions(precision=2)

clusters = 3
n_cluster = [90, 50, 75]
n_total = sum(n_cluster)
means = [9, 21, 35]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
sns.kdeplot(np.array(mix))
plt.xlabel('$x$', fontsize=14)
plt.savefig('img701.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

import matplotlib.tri as tri
from functools import reduce
from matplotlib import ticker, cm

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0 for i in range(3)]

def xy2bc(xy, tol=1.e-3):
  '''Converts 2D Cartesian coordinates to barycentric.
  Arguments:
    xy: A length-2 sequence containing the x and y value.
  '''
  s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75 for i in range(3)]
  return np.clip(s, tol, 1.0 - tol)

class Dirichlet(object):
  def __init__(self, alpha):
    '''Creates Dirichlet distribution with parameter `alpha`.'''
    from math import gamma
    from operator import mul
    self._alpha = np.array(alpha)
    self._coef = gamma(np.sum(self._alpha)) /reduce(mul, [gamma(a) for a in self._alpha])
  def pdf(self, x):
    '''Returns pdf value for `x`.'''
    from operator import mul
    return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])
  def sample(self, N):
    '''Generates a random sample of size `N`.'''
    return np.random.dirichlet(self._alpha, N)

def draw_pdf_contours(dist, nlevels=100, subdiv=8, **kwargs):
  '''Draws pdf contours over an equilateral triangle (2-simplex).
  Arguments:
    dist: A distribution instance with a `pdf` method.
    border (bool): If True, the simplex border is drawn.
    nlevels (int): Number of contours to draw.
    subdiv (int): Number of recursive mesh subdivisions to create.
    kwargs: Keyword args passed on to `plt.triplot`.
  '''
  refiner = tri.UniformTriRefiner(_triangle)
  trimesh = refiner.refine_triangulation(subdiv=subdiv)
  pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

  plt.tricontourf(trimesh, pvals, nlevels, cmap=cm.Blues, **kwargs)
  plt.axis('equal')
  plt.xlim(0, 1)
  plt.ylim(0, 0.75**0.5)
  plt.axis('off')

alphas = [[0.5] * 3, [1] * 3, [10] * 3, [2, 5, 10]]
for (i, alpha) in enumerate(alphas):
  plt.subplot(2, 2, i + 1)
  dist = Dirichlet(alpha)
  draw_pdf_contours(dist)
  plt.title(r'$\alpha$ = ({:.1f}, {:.1f}, {:.1f})'.format(*alpha), fontsize=16)
  
plt.savefig('img702.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_kg:
  p = pm.Dirichlet('p', a=np.ones(clusters))
  category = pm.Categorical('category', p=p, shape=n_total) 
  
  means = pm.math.constant([10, 20, 35])
  y = pm.Normal('y', mu=means[category], sd=2, observed=mix)

  trace_kg = pm.sample(10000, njobs=1)

chain_kg = trace_kg[1000:]
varnames_kg = ['p']
pm.traceplot(chain_kg, varnames_kg)
plt.savefig('img704.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_ug:
  p = pm.Dirichlet('p', a=np.ones(clusters))
  category = pm.Categorical('category', p=p, shape=n_total) 

  means = pm.Normal('means', mu=[10, 20, 35], sd=2, shape=clusters)
  sd = pm.HalfCauchy('sd', 5)
  y = pm.Normal('y', mu=means[category], sd=sd, observed=mix)

  trace_ug = pm.sample(10000, njobs=1)

chain_ug = trace_ug[1000:]
varnames_ug = ['means', 'sd', 'p']
pm.traceplot(chain_ug, varnames_ug)
plt.savefig('img705.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

pm.summary(chain_ug, varnames_ug)

ppc = pm.sample_ppc(chain_ug, 50, model_ug)
for i in ppc['y']:
  sns.kdeplot(i, alpha=0.1, color='b')

sns.kdeplot(np.array(mix), lw=2, color='k')
plt.xlabel('$x$', fontsize=14)
plt.savefig('img706.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_mg:
  p = pm.Dirichlet('p', a=np.ones(clusters))

  means = pm.Normal('means', mu=[10, 20, 35], sd=2, shape=clusters)
  sd = pm.HalfCauchy('sd', 5)
  
  y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)

  trace_mg = pm.sample(5000, njobs=1)

chain_mg = trace_mg[500:]
varnames_mg = ['means', 'sd', 'p']
pm.traceplot(chain_mg, varnames_mg);
plt.savefig('img7exercise.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

lam_params = [0.5, 1.5, 3, 8]
k = np.arange(0, max(lam_params) * 3)
for lam in lam_params:
  y = stats.poisson(lam).pmf(k)
  plt.plot(k, y, 'o-', label="$\\lambda$ = {:3.1f}".format(lam))
plt.legend()
plt.xlabel('$k$', fontsize=14)
plt.ylabel('$pmf(k)$', fontsize=14)
plt.savefig('img707.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

np.random.seed(42)
n = 100
theta = 2.5
pi = 0.1

counts = np.array([(np.random.random() > pi) * np.random.poisson(theta) for i in range(n)])

with pm.Model() as ZIP:
  psi = pm.Beta('psi', 1, 1)
  lam = pm.Gamma('lam', 2, 0.1)
  
  y = pm.ZeroInflatedPoisson('y', psi, lam, observed=counts)
  trace_ZIP = pm.sample(5000, njobs=1)

chain_ZIP = trace_ZIP[100:]
pm.traceplot(chain_ZIP);
plt.savefig('img708.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

#https://stats.idre.ucla.edu/stat/data/fish.csv
fish_data = pd.read_csv('fish.csv')
fish_data.head()

with pm.Model() as ZIP_reg:
  psi = pm.Beta('psi', 1, 1)
  
  alpha = pm.Normal('alpha', 0, 10)
  beta = pm.Normal('beta', 0, 10, shape=2)
  lam = pm.math.exp(alpha + beta[0] * fish_data['child'] + beta[1] * fish_data['camper'])
  
  y = pm.ZeroInflatedPoisson('y', psi, lam, observed=fish_data['count'])
  trace_ZIP_reg = pm.sample(2000, njobs=1)

chain_ZIP_reg = trace_ZIP_reg[100:]
pm.traceplot(chain_ZIP_reg);
plt.savefig('img710.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

children =  [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
thin = 5
for n in children:
  without_camper = chain_ZIP_reg['alpha'][::thin] + chain_ZIP_reg['beta'][:,0][::thin] * n
  with_camper = without_camper + chain_ZIP_reg['beta'][:,1][::thin]
  fish_count_pred_0.append(np.exp(without_camper))
  fish_count_pred_1.append(np.exp(with_camper))

plt.plot(children, fish_count_pred_0, 'bo', alpha=0.01)
plt.plot(children, fish_count_pred_1, 'ro', alpha=0.01)

plt.xticks(children);
plt.xlabel('Number of children', fontsize=14)
plt.ylabel('Fish caught', fontsize=14)
plt.plot([], 'bo', label='without camper')
plt.plot([], 'ro', label='with camper')
plt.legend(fontsize=14)
plt.savefig('img711.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length' 
x_0 = df[x_n].values
y_0 = np.concatenate((y_0, np.ones(6)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
x_0_m = x_0 - x_0.mean()
plt.plot(x_0, y_0, 'o', color='k')
plt.savefig('img712.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_rlg:
  alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=100)
  beta = pm.Normal('beta', mu=0, sd=10)
  mu = alpha_tmp + beta * x_0_m
  theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
  
  pi = pm.Beta('pi', 1, 1)
  p = pi * 0.5 + (1 - pi) * theta
  
  alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_0.mean())
  bd = pm.Deterministic('bd', -alpha/beta)
  
  yl = pm.Bernoulli('yl', p=p, observed=y_0)
  trace_rlg = pm.sample(2000, njobs=1)

varnames = ['alpha', 'beta', 'bd', 'pi']
pm.traceplot(trace_rlg, varnames)
plt.savefig('img713.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

pm.summary(trace_rlg, varnames)

theta = trace_rlg['theta'].mean(axis=0)
idx = np.argsort(x_0)
plt.plot(x_0[idx], theta[idx], color='b', lw=3);
plt.axvline(trace_rlg['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(trace_rlg['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

plt.plot(x_0, y_0, 'o', color='k')
theta_hpd = pm.hpd(trace_rlg['theta'])[idx]
plt.fill_between(x_0[idx], theta_hpd[:,0], theta_hpd[:,1], color='b', alpha=0.5)

plt.xlabel(x_n, fontsize=16)
plt.ylabel('$\\theta$', rotation=0, fontsize=16)
plt.savefig('img714.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

