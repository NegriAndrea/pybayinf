#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import matplotlib as mpl
# mpl.use('Agg')
# mpl.rc('text', usetex=True)
# mpl.rc('font', family='serif', size=11)
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
os.environ["OMP_NUM_THREADS"] = "1"


import scipy.stats
import scipy.special
Lstar = 1. #just a number
alpha = 2.
k = alpha +1
# x = np.random.gamma(k, scale=Lstar, size=20000)
x = scipy.stats.gamma.rvs(k, scale=Lstar, size=20000)

be = np.logspace(-6, 1, 500)

data, be = np.histogram(x, be)
total_data = np.sum(data)

fig=plt.figure()
ax=fig.add_subplot(111)
dataerr = np.sqrt(data)/np.diff(be)/total_data
datanorm = data/np.diff(be)/total_data
ax.errorbar(be[0:-1], datanorm, yerr=dataerr,xerr=np.diff(be), fmt='o')


# the real function
x_real = np.logspace(-6., 1, 100)
realfunc = scipy.stats.gamma.pdf(x_real, k,
        scale=Lstar)
ax.plot(x_real, realfunc)
# print(realfunc)

ax.set_yscale('log')
ax.set_xscale('log')
# ax.set_xlim([1e43, 2e45])
ax.set_xlim([-0.01, 10])

fig.savefig('tmp.pdf')



# def log_likelihood(theta, x, y, yerr):
    # Sigma, Mu,  log_f = theta
    # model = 1./np.sqrt(2.*np.pi*Sigma**2) * np.exp((mu-x)**2/(2.*Sigma**2))
    # sigma2 = yerr**2 + model**2*np.exp(2*log_f)
    # return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


def ln_likelihood(theta, x):

    k, Lstar = theta
    n=x.size

    return ((k-1)*np.sum(np.log(x)) - 1./Lstar*np.sum(x) - n*k*np.log(Lstar) -
            n*np.log(scipy.special.gamma(k)))


def ln_prior(theta):

    k, Lstar = theta

    if 0.< k < 4. and 0.1 < Lstar < 2.:
        return 0.

    return -np.inf


def ln_probability(theta, x):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta, x)


from scipy.optimize import minimize
np.random.seed(42)
nll = lambda *args: -ln_likelihood(*args)
initial = np.array([1., 0.1]) + 0.1*np.random.randn(2)
soln = minimize(nll, initial, args=(x))
print("Maximum likelihood estimates:")
print('alpha= ', soln.x[0]-1)
print('Lstar= ', '{:e}'.format(soln.x[1]))

pos = soln.x + 1e-2*np.random.randn(100, 2)
print (pos )
nwalkers, ndim = pos.shape
print(nwalkers, ndim )

import emcee
from multiprocessing import Pool
from contextlib import closing
# with Pool() as pool:
with closing(Pool(processes=4)) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, 
            ln_probability, args=(x,), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)
    pool.terminate()


fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["k", "Lstar"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    # ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

fig.savefig('chains.pdf')



flat_samples = sampler.get_chain(discard=1000, thin=20, flat=True)
import corner
fig = corner.corner(flat_samples, labels=labels)
fig.savefig('cornerplot.pdf')
