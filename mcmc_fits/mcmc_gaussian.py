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
from scipy.optimize import curve_fit
from numba import njit
import time
os.environ["OMP_NUM_THREADS"] = "1"


@njit
def get_samples(x, chain):
    N = chain.shape[0]
    samples = np.zeros((N, x.size))
    for i in range(N):
        samples[i,:] = gaussianmine(x, chain[i,0], chain[i, 1])

    return samples

def calc_shaded_region(chain, x_plot):
    """
    Calculate the shaded region for 1 sigma values in the fit by using the
    chains in the mcmc.

    """
    x = np.asarray(x_plot)
    y_max  = np.full_like(x, -np.inf)
    y_min  = np.full_like(x,  np.inf)

    t = time.time()
    samples = get_samples(x, chain)
    print('loop time ', time.time()-t)

    # c0 = np.tile(chain[:,0],[x.size,1]).T
    # c1 = np.tile(chain[:,1],[x.size,1]).T
    # x2 = np.broadcast_to(x,(chain.shape[0],x.size))
    # samples = gaussianmine(x2, c0, c1)

    t = time.time()
    y_max = np.quantile(samples, 0.16, axis=0)
    y_min = np.quantile(samples, 0.84, axis=0)
    median = np.quantile(samples, 0.5, axis=0)
    print('quantile time ', time.time()-t)

    return y_max, y_min, median

x = np.random.normal(size=100)

data, be = np.histogram(x, 20)

fig=plt.figure()
ax=fig.add_subplot(111)
dataerr = np.sqrt(data)/(be[1]-be[0])/np.sum(data)
datanorm = data/(be[1]-be[0])/np.sum(data)
ax.errorbar((be[:-1]+be[1:])/2., datanorm, yerr=dataerr,xerr=np.diff(be), fmt='o')

@njit
def gaussianmine(x, sigma, mu):
    return 1./(np.sqrt(2.*np.pi)*sigma) * np.exp(-((x-mu)/sigma)**2/2.)

# def gauss(x, *p):
    # A, mu, sigma = p
    # return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1.,0.]

coeff, var_matrix = curve_fit(gaussianmine, (be[:-1]+be[1:])/2., datanorm,
        sigma=dataerr, p0 = p0) #, bounds = ([0., -10.],[10.,10]))
errs = np.sqrt(np.diag(var_matrix))
print('Histogram fit')
print('sigma=', coeff[0], errs[0])
print('mu=', coeff[1], errs[1])


# the real function
x_real = np.linspace(-3, 3, 1000)
realfunc = 1./np.sqrt(2.*np.pi) * np.exp(-x_real**2/2.)
ax.plot(x_real, realfunc)

ax.set_yscale('log')



# def log_likelihood(theta, x, y, yerr):
    # Sigma, Mu,  log_f = theta
    # model = 1./np.sqrt(2.*np.pi*Sigma**2) * np.exp((mu-x)**2/(2.*Sigma**2))
    # sigma2 = yerr**2 + model**2*np.exp(2*log_f)
    # return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


def ln_likelihood(theta, x):

    Sigma, Mu = theta
    n=np.size(x)

    return (n/2.*np.log(2.*np.pi) - n/2.*np.log(Sigma**2) -
            1./(2.*Sigma**2)*np.sum((x-Mu)**2))


def ln_prior(theta):

    Sigma, Mu = theta

    if 0.8<np.abs(Sigma) < 1.2 and -1.< Mu < 1.:
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
print('sigma= ', soln.x[0])
print('mu= ', soln.x[1])

pos = soln.x + 1e-2*np.random.randn(100, 2)
nwalkers, ndim = pos.shape
print(nwalkers, ndim)

import emcee
from multiprocessing import Pool
from contextlib import closing
# with Pool() as pool:
with closing(Pool(processes=4)) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, 
            ln_probability, args=(x,), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)
    pool.terminate()


fig2, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Sigma", "mu"]
for i in range(ndim):
    axnew = axes[i]
    axnew.plot(samples[:, :, i], "k", alpha=0.3)
    axnew.set_xlim(0, len(samples))
    axnew.set_ylabel(labels[i])
    axnew.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

fig2.savefig('chains.pdf')



flat_samples = sampler.get_chain(discard=1000, thin=1, flat=True)
import corner
fig3 = corner.corner(flat_samples, labels=labels, 
        levels=1.0 - np.exp(-0.5 * np.array([0.5, 1., 2.]) ** 2))
fig3.savefig('cornerplot.pdf')

sigma =np.quantile(flat_samples[:,0], [0.16,0.5,.84])
mu = np.quantile(flat_samples[:,1], [0.16,0.5,.84])


print('sigma = ', sigma[1], ' ', sigma[1]-sigma[2], '  + ', sigma[1]-sigma[0])
print('mu = ', mu[1], ' ', mu[1]-mu[2], '  + ', mu[1]-mu[0])

y_max, y_min, median = calc_shaded_region(flat_samples, x_real)
ax.fill_between(x_real, y_min, y_max, facecolor='r', alpha=0.2)
ax.plot(x_real, median, color='r')

# single, pcov = curve_fit(f_single, x, y, sigma=yerr, bounds=([-5.0, 0.0, -26.0], [3.0, 2.5, -18.0]))

fig.savefig('tmp.pdf')
