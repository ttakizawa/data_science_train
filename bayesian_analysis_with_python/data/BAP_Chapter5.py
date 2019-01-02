#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 5
##################################################################

import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
plt.style.use('seaborn-darkgrid')
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

z = np.linspace(-10, 10, 100)
logistic = 1 / (1 + np.exp(-z))
plt.plot(z, logistic)
plt.xlabel('$z$', fontsize=18)
plt.ylabel('$logistic(z)$', fontsize=18)
plt.savefig('img501.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

iris = sns.load_dataset('iris')
iris.head()
sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True)
plt.savefig('img503.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

sns.pairplot(iris, hue='species', diag_kind='kde')
plt.savefig('img504.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length' 
x_0 = df[x_n].values

with pm.Model() as model_0:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=10)
  
  mu = alpha + pm.math.dot(x_0, beta)
  theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
  bd = pm.Deterministic('bd', -alpha/beta)
  yl = pm.Bernoulli('yl', p=theta, observed=y_0)

  trace_0 = pm.sample(5000)

chain_0 = trace_0[1000:]
varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_0, varnames)
plt.savefig('img505.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.summary(chain_0, varnames)

theta = chain_0['theta'].mean(axis=0)
idx = np.argsort(x_0)
plt.plot(x_0[idx], theta[idx], color='b', lw=3);
plt.axvline(chain_0['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(chain_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

plt.plot(x_0, y_0, 'o', color='k')
theta_hpd = pm.hpd(chain_0['theta'])[idx]
plt.fill_between(x_0[idx], theta_hpd[:,0], theta_hpd[:,1], color='b', alpha=0.5)

plt.xlabel(x_n, fontsize=16)
plt.ylabel(r'$\theta$', rotation=0, fontsize=16)
plt.savefig('img506.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

def classify(n, threshold):
  """
  A simple classifying function
  """
  n = np.array(n)
  mu = chain_0['alpha'].mean() + chain_0['beta'].mean() * n
  prob = 1 / (1 + np.exp(-mu))
  return prob, prob >= threshold

classify([5, 5.5, 6], 0.5)

df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_1 = df[x_n].values

with pm.Model() as model_1:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))
  
  mu = alpha + pm.math.dot(x_1, beta)
  theta = 1 / (1 + pm.math.exp(-mu))
  bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0])
  yl = pm.Bernoulli('yl', p=theta, observed=y_1)
  
  trace_1 = pm.sample(5000, njobs=1)

chain_1 = trace_1[1000:]
varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_1, varnames)
plt.savefig('img507.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

idx = np.argsort(x_1[:,0])
ld = chain_1['bd'].mean(0)[idx]

plt.scatter(x_1[:,0], x_1[:,1], c=y_0, cmap='viridis')
plt.plot(x_1[:,0][idx], ld, color='r');

ld_hpd = pm.hpd(chain_1['bd'])[idx]
plt.fill_between(x_1[:,0][idx], ld_hpd[:,0], ld_hpd[:,1], color='r', alpha=0.5);

plt.xlabel(x_n[0], fontsize=16)
plt.ylabel(x_n[1], fontsize=16)
plt.savefig('img508.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

corr = iris[iris['species'] != 'virginica'].corr()
mask = np.tri(*corr.shape).T
sns.heatmap(corr.abs(), mask=mask, annot=True, cmap='viridis')
plt.savefig('img509.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

df = iris.query("species == ('setosa', 'versicolor')")
df = df[45:] #df[22:78]
y_3 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width'] 
x_3 = df[x_n].values

with pm.Model() as model_3:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))
  
  mu = alpha + pm.math.dot(x_3, beta)
  p = 1 / (1 + pm.math.exp(-mu))
  ld = pm.Deterministic('ld', -alpha/beta[1] - beta[0]/beta[1] * x_3[:,0])
  yl = pm.Bernoulli('yl', p=p, observed=y_3)

  trace_3 = pm.sample(5000)

cadena_3 = trace_3[:]
varnames = ['alpha', 'beta']
pm.traceplot(cadena_3, varnames);
plt.savefig('img510-0.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

idx = np.argsort(x_3[:,0])
ld = trace_3['ld'].mean(0)[idx]
plt.scatter(x_3[:,0], x_3[:,1], c=y_3, cmap='viridis')
plt.plot(x_3[:,0][idx], ld, color='r');

ld_hpd = pm.hpd(trace_3['ld'])[idx]
plt.fill_between(x_3[:,0][idx], ld_hpd[:,0], ld_hpd[:,1], color='r', alpha=0.5);

plt.xlabel(x_n[0], fontsize=16)
plt.ylabel(x_n[1], fontsize=16)
plt.savefig('img510.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

iris = sns.load_dataset('iris')
y_s = pd.Categorical(iris['species']).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values
x_s = (x_s - x_s.mean(axis=0))/x_s.std(axis=0)

with pm.Model() as model_s:
  alpha = pm.Normal('alpha', mu=0, sd=2, shape=3)
  beta = pm.Normal('beta', mu=0, sd=2, shape=(4,3))
  
  mu = alpha + pm.math.dot(x_s, beta)
  theta = tt.nnet.softmax(mu)
  
  yl = pm.Categorical('yl', p=theta, observed=y_s)
  trace_s = pm.sample(2000, njobs=1)

pm.traceplot(trace_s)
plt.savefig('img512.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

data_pred = trace_s['alpha'].mean(axis=0) + np.dot(x_s, trace_s['beta'].mean(axis=0))
y_pred = []
for point in data_pred:
  y_pred.append(np.exp(point)/np.sum(np.exp(point), axis=0))
np.sum(y_s == np.argmax(y_pred, axis=1))/len(y_s)

with pm.Model() as model_sf:
  alpha = pm.Normal('alpha', mu=0, sd=2, shape=2)
  beta = pm.Normal('beta', mu=0, sd=2, shape=(4,2))
  
  alpha_f = tt.concatenate([[0] , alpha])
  beta_f = tt.concatenate([np.zeros((4,1)) , beta], axis=1)
  
  mu = alpha_f + pm.math.dot(x_s, beta_f)
  theta = tt.nnet.softmax(mu)
  
  yl = pm.Categorical('yl', p=theta, observed=y_s)
  trace_sf = pm.sample(5000, njobs=1)

pm.traceplot(trace_sf)
plt.savefig('img513.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length' 
x_0 = df[x_n].values

with pm.Model() as model_lda:
  mus = pm.Normal('mus', mu=0, sd=10, shape=2)
  sigma = pm.Uniform('sigma', 0, 10)
  setosa = pm.Normal('setosa', mu=mus[0], sd=sigma, observed=x_0[:50])
  versicolor = pm.Normal('versicolor', mu=mus[1], sd=sigma, observed=x_0[50:])
  bd = pm.Deterministic('bd', (mus[0]+mus[1])/2)
  
  trace_lda = pm.sample(5200, njobs=1)

chain_lda = trace_lda[200:]
pm.traceplot(chain_lda)
plt.savefig('img514.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

x_n = 'sepal_length' 
plt.axvline(trace_lda['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(trace_lda['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)
plt.plot(x_0, y_0, 'o', color='k')
plt.xlabel(x_n, fontsize=16)
plt.savefig('img515.png', dpi=300, figsize=(5.5, 5.5))

plt.figure()

pm.summary(trace_lda)

