#
# Bayesian Analysis with Python written by Osvaldo Marthin
# Chapter 6
##################################################################

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

x = np.array([4.,5.,6.,9.,12, 14.])
y = np.array([4.2, 6., 6., 9., 10, 10.])

order = [0, 1, 2, 5]
plt.plot(x, y, 'o')
for i in order:
  x_n = np.linspace(x.min(), x.max(), 100)
  coeffs = np.polyfit(x, y, deg=i)
  ffit = np.polyval(coeffs, x_n)
  
  p = np.poly1d(coeffs)
  yhat = p(x)
  ybar = np.mean(y)
  ssreg = np.sum((yhat-ybar)**2)
  sstot = np.sum((y - ybar)**2) 
  r2 = ssreg / sstot
  
  plt.plot(x_n, ffit, label='order {}, $R^2$= {:.2f}'.format(i, r2))

plt.legend(loc=2, fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14, rotation=0)
plt.savefig('img601.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

x1 = np.array([10,7])
y1 = np.array([9,7])

plt.plot(x, y, 'o')
plt.plot(x1,y1, 's')
for i in order:
  x_n = np.linspace(x.min(), x.max(), 100)
  coeffs = np.polyfit(x, y, deg=i)
  ffit = np.polyval(coeffs, x_n)
  
  p = np.poly1d(coeffs)
  yhat = p(x)
  ybar = np.mean(y)
  ssreg = np.sum((yhat-ybar)**2)
  sstot = np.sum((y - ybar)**2) 
  r2 = ssreg / sstot
  
  plt.plot(x_n, ffit, label='order {}, $R^2$= {:.2f}'.format(i, r2))

plt.legend(loc=2, fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14, rotation=0)
plt.savefig('img602.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

plt.figure(figsize=(8, 6))
x_values = np.linspace(-10, 10, 300)
for df in [1, 2, 5, 15]:
  distri = stats.laplace(scale=df)
  x_pdf = distri.pdf(x_values)
  plt.plot(x_values, x_pdf, label='$b$ = {}'.format(df))

x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, label='Gaussian')
plt.xlabel('x')
plt.ylabel('p(x)', rotation=0)
plt.legend(loc=0, fontsize=14)
plt.xlim(-7, 7);
plt.savefig('img603.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

x_1 = np.array([ 10.,   8.,  13.,   9.,  11.,  14.,   6.,   4.,  12.,   7.,   5.])
y_1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26,10.84, 4.82, 5.68])

np.random.seed(1)
real_alpha = 4.25
real_beta = [8.7, -1.2]
data_size = 20
noise = np.random.normal(0, 2, size=data_size)
x_1 = np.linspace(0, 5, data_size)
y_1 = real_alpha + real_beta[0] * x_1 + real_beta[1] * x_1**2 + noise

order = 2
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean())/y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img604.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_l:
  alpha = pm.Normal('alpha', mu=0, sd=10)
  beta = pm.Normal('beta', mu=0, sd=1)
  epsilon = pm.HalfCauchy('epsilon', 5)
  mu = alpha + beta * x_1s[0]
  y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
  
  trace_l = pm.sample(2100, njobs=1)

chain_l = trace_l[100:]
pm.traceplot(chain_l);

pm.summary(chain_l)

with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=1, shape=x_1s.shape[0])  
    epsilon = pm.HalfCauchy('epsilon', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
    trace_p = pm.sample(2100, njobs=1)

chain_p = trace_p[100:]
pm.traceplot(chain_p);

pm.summary(chain_p)

plt.figure()

alpha_l_post = chain_l['alpha'].mean()
betas_l_post = chain_l['beta'].mean(axis=0)
idx = np.argsort(x_1s[0])
y_l_post = alpha_l_post + betas_l_post * x_1s[0]

plt.plot(x_1s[0][idx], y_l_post[idx], label='Linear')

alpha_p_post = chain_p['alpha'].mean()
betas_p_post = chain_p['beta'].mean(axis=0)
y_p_post = alpha_p_post + np.dot(betas_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], label='Pol order {}'.format(order))

plt.scatter(x_1s[0], y_1s)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0);
plt.legend()
plt.savefig('img605.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

#dic_l = pm.dic(trace=trace_l, model=model_l)
#dic_l
#dic_p = pm.dic(trace=trace_p, model=model_p)
#dic_p
waic_l = pm.waic(trace=trace_l, model=model_l)
waic_l
waic_p = pm.waic(trace=trace_p, model=model_p)
waic_p
loo_l = pm.loo(trace=trace_l, model=model_l)
loo_l
loo_p = pm.loo(trace=trace_p, model=model_p)
loo_p

plt.figure(figsize=(8, 4))
plt.subplot(121)
for idx, ic in enumerate((waic_l, waic_p)):
  plt.errorbar(ic[0], idx, xerr=ic[1], fmt='bo')
plt.title('WAIC')
plt.yticks([0, 1], ['linear', 'quadratic'])
plt.ylim(-1, 2)

plt.subplot(122)
for idx, ic in enumerate((loo_l, loo_p)):
  plt.errorbar(ic[0], idx, xerr=ic[1], fmt='go')
plt.title('LOO')
plt.yticks([0, 1], ['linear', 'quadratic'])
plt.ylim(-1, 2)
plt.tight_layout()
plt.savefig('img606.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.scatter(x_1s[0], y_1s, c='r');
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Linear')

for i in range(0, len(chain_l['alpha']), 50):
  plt.scatter(x_1s[0], chain_l['alpha'][i] + chain_l['beta'][i]*x_1s[0], c='g', edgecolors='g', alpha=0.5);
plt.plot(x_1s[0], chain_l['alpha'].mean() + chain_l['beta'].mean()*x_1s[0], c='g', alpha=1)

plt.subplot(122)
plt.scatter(x_1s[0], y_1s, c='r');
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Order {}'.format(order))
for i in range(0, len(chain_p['alpha']), 50):
  plt.scatter(x_1s[0], chain_p['alpha'][i] + np.dot(chain_p['beta'][i], x_1s), c='g', edgecolors='g', alpha=0.5)
idx = np.argsort(x_1)
plt.plot(x_1s[0][idx], alpha_p_post + np.dot(betas_p_post, x_1s)[idx], c='g', alpha=1);
plt.savefig('img607.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

coins = 30 # 300
heads = 9 # 90
y_d = np.repeat([0, 1], [coins-heads, heads])

with pm.Model() as model_BF:
  p = np.array([0.5, 0.5])
  model_index = pm.Categorical('model_index', p=p)
  m_0 = (4, 8)
  m_1 = (8, 4)
  m = pm.math.switch(pm.math.eq(model_index, 0), m_0, m_1)
  
  theta = pm.Beta('theta', m[0], m[1])
  y = pm.Bernoulli('y', theta, observed=y_d)
  trace_BF = pm.sample(5500, njobs=1)

chain_BF = trace_BF[500:]
pm.traceplot(chain_BF)
plt.savefig('img609.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

pM1 = chain_BF['model_index'].mean()
pM0 = 1 - pM1
BF = (pM0/pM1)*(p[1]/p[0])
print(pM0, pM1, BF)

with pm.Model() as model_BF_0:
  theta = pm.Beta('theta', 4, 8)
  y = pm.Bernoulli('y', theta, observed=y_d)
  trace_BF_0 = pm.sample(5500, njobs=1)  

chain_BF_0 = trace_BF_0[500:]
pm.traceplot(chain_BF_0)
plt.savefig('img610.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

with pm.Model() as model_BF_1:
  theta = pm.Beta('theta', 8, 4)
  y = pm.Bernoulli('y', theta, observed=y_d)

  trace_BF_1 = pm.sample(5500, njobs=1)

chain_BF_1 = trace_BF_1[500:]
pm.traceplot(chain_BF_1)
plt.savefig('img611.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

est = [((38.02, 4.17), (39.41, 2.04)), ((36.69, 3.96), (38.09, 1.94)),
       ((368.41, 13.40), (368.76, 12.48)) , ((366.61, 13.31), (366.87, 12.34))]

title = ['WAIC 30_9', 'LOO 30_9', 'WAIC 300_90', 'LOO 300_90']

for i in range(4):
  plt.subplot(2,2,i+1)
  for idx, ic in enumerate(est[i]):
    plt.errorbar(ic[0], idx, xerr=ic[1], fmt='bo')
  plt.title(title[i])
  plt.yticks([0, 1], ['model_0', 'model_1'])
  plt.ylim(-1, 2)

plt.savefig('img612.png', dpi=300, figsize=[5.5, 5.5])

plt.figure()

