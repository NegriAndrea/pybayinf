#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')
# mpl.rc('text', usetex=True)
# mpl.rc('font', family='serif', size=11)
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
os.environ["OMP_NUM_THREADS"] = "1"



def _schechter(M, alpha, phi, Mo):
    import numpy as np
    f = phi * 10.0**(0.4 * (alpha + 1) * (Mo - M))
    out = 0.4 * np.log(10.0) * np.exp(-10.0**(0.4 * (Mo - M))) * f
    return out

from scipy import stats
from scipy.integrate import quad
class sch(stats.rv_continuous):
    # def __init__(self):
        # self.norm = quad(_schechter, self.a, self.b, 
                # args=(alpha, phi, Mo))
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (np.asarray(arg) > -np.inf))
        return cond

    def _pdf(self, x, alpha, phi, Mo):
        return np.asarray(_schechter(x, alpha, phi, Mo))/self.norm

    def __init__(self, alpha=0., phi=1., Mo=-21., *args, **kwargs):
        super(sch, self).__init__(self, *args, **kwargs)

        # calculate the normalization
        self.norm = quad(_schechter, self.a, self.b, 
                args=(alpha, phi, Mo))[0]

    # def norm(self, alpha, phi, Mo):
        # self.norm = quad(_schechter, self.a, self.b, 
                # args=(alpha, phi, Mo))[0]


# to normalize the distribution, it has to be limited on the right, b cannot be
# np.inf
alpha = -1.4
phi= 38000.
Mo=-21.78
Mlim_left = -15.
dist = sch(name='schecter',a=-100., b=Mlim_left, shapes='alpha, phi, Mo', alpha=alpha,
        phi=phi, Mo=Mo)



# yy= dist.pdf(x_real, alpha=alpha, phi=phi, Mo=Mo)
# plt.semilogy(x_real, yy)

filename = 'sch_sampling.h5'
generate = True
if generate:
    x5 = dist.rvs(alpha=alpha, phi=phi, Mo=Mo, size=5)
    x10 = dist.rvs(alpha=alpha, phi=phi, Mo=Mo, size=10)
    x100 = dist.rvs(alpha=alpha, phi=phi, Mo=Mo, size=100)
    x1000 = dist.rvs(alpha=alpha, phi=phi, Mo=Mo, size=1000)
    with h5py.File(filename, 'w') as ff:
        ff['x5'] = x5
        ff['x10'] = x10
        ff['x100'] = x100
        ff['x1000'] = x1000
else:
    with h5py.File(filename, 'r') as ff:
        x = ff['x'][()]

exit()

#--------------------------------------------------------
filename = '/net/deimos/scratch1/anegri/EAGLE/L0100N1504/REFERENCE/data/dataMagnitudes_030kpc_EMILES_PDXX_DUST_CH_028_z000p000.hdf5'
# filename = 'dataMagnitudes_030kpc_EMILES_PDXX_DUST_CH_028_z000p000.hdf5'
with h5py.File(filename, 'r') as ff:
    r = ff['/Data/r-Magnitude'][()]
    Mstar = ff['/Data/StellarMass'][()]
    SFR = ff['/Data/SFR'][()]
    sSFR = SFR/Mstar
x = r[(r<Mlim_left) & (sSFR < 1e-11)]
x = r[r<Mlim_left]

#--------------------------------------------------------
print('read')

be = np.linspace(-28, Mlim_left, 70)

data, be = np.histogram(x, be)
total_data = np.sum(data)

fig=plt.figure()
ax=fig.add_subplot(111)
dataerr = np.sqrt(data)/np.diff(be)/total_data
# datanorm = data/np.diff(be)/total_data
datanorm = data/np.diff(be)
ax.errorbar(be[0:-1], datanorm, yerr=dataerr,xerr=np.diff(be), fmt='o')
del dataerr, datanorm, total_data


# the real function
x_real = np.linspace(-30, 10, 1000)
realfunc = dist.pdf(x_real, alpha=alpha, phi=phi, Mo=Mo)
ax.plot(x_real, realfunc)
del realfunc, x_real
# print(realfunc)

ax.set_yscale('log')
ax.set_ylim([1e-17, 1.e10])
ax.set_xlim([-30, Mlim_left])

fig.savefig('tmp.pdf')


def ln_likelihood(theta, x):

    alpha, phi, Mo, alphaF, phiF, MoF = theta
    n=x.size

    out = ( -quad(_schechter, -100., Mlim_left, args=(alpha , phi , Mo ))[0]
            -quad(_schechter, -100., Mlim_left, args=(alphaF, phiF, MoF))[0]

            + np.sum(np.log(_schechter(x, alpha=alpha , phi=phi , Mo=Mo )
            + _schechter(x, alpha=alphaF, phi=phiF, Mo=MoF)))
            )

    return out



def ln_prior(theta):

    alpha, phi, Mo, alphaF, phiF, MoF = theta

    if -2.5 < alpha<2. and -25. < Mo < -20. and 0. < phi < 1.e5 \
            and MoF > Mo + 2. and -2.5 < alphaF <0. and  0. < phiF < 1.e5:
        return 0.

    return -np.inf


def ln_probability(theta, x):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta, x)

# from scipy.optimize import minimize
# np.random.seed(42)
# nll = lambda *args: -ln_likelihood(*args)
# initial = np.array([alpha, phi, Mo]) # + 0.01*np.random.randn(3)
# soln = minimize(nll, initial, args=(x))
# print("Maximum likelihood estimates:")
# print('alpha= ', soln.x[0])
# # print('phi = ', '{:e}'.format(soln.x[1]))
# print('phi = ', soln.x[1])
# print('Mo = ', soln.x[2])



# pos = soln.x + 1e-2*np.random.randn(50, 3)
initial = np.array([alpha, phi, Mo, alpha, phi, Mo+2.5])
pos = initial + 1e-2*np.random.randn(100, 6)
print(pos)
nwalkers, ndim = pos.shape
print(nwalkers, ndim )

import emcee
from multiprocessing import Pool
from contextlib import closing
# with Pool() as pool:
with closing(Pool(processes=8)) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, 
            ln_probability, args=(x,), pool=pool)
    sampler.run_mcmc(pos, 5000, progress=True)
    pool.terminate()


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [r'$\alpha$', r'$\phi$', r'$M_\star$',r'$\alpha_f$', r'$\phi_f$',
        r'$M_{\star,f}$']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    # ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

fig.savefig('chains.pdf')



flat_samples = sampler.get_chain(discard=1500, thin=1, flat=True)
import corner
fig = corner.corner(flat_samples, labels=labels, show_titles=True)
fig.savefig('cornerplot.pdf')

alphamcmc= np.median(flat_samples[:,0])
phimcmc= np.median(flat_samples[:,1])
Momcmc= np.median(flat_samples[:,2])

alphaFmcmc= np.median(flat_samples[:,3])
phiFmcmc= np.median(flat_samples[:,4])
MoFmcmc= np.median(flat_samples[:,5])

print('alpha=', alphamcmc)
print('phi=', phimcmc)
print('Mo=', Momcmc)

be = np.linspace(-28, Mlim_left, 70)

data, be = np.histogram(x, be)
# total_data = np.sum(data)

fig=plt.figure('realfig')
ax=fig.add_subplot(111)
dataerr = np.sqrt(data)/np.diff(be)
# datanorm = data/np.diff(be)/total_data
datanorm = data/np.diff(be)
ax.errorbar(be[0:-1], datanorm, yerr=dataerr,xerr=np.diff(be), fmt='o')
del dataerr, datanorm


# the real function
x_real = np.linspace(-30, 10, 1000)
realfunc = (_schechter(x_real, alpha=alphamcmc, phi=phimcmc, Mo=Momcmc) +
        _schechter(x_real, alpha=alphaFmcmc, phi=phiFmcmc, Mo=MoFmcmc))

ax.plot(x_real, realfunc, c='k')
ax.plot(x_real,_schechter(x_real, alpha=alphamcmc, phi=phimcmc, Mo=Momcmc),
        c='r')
ax.plot(x_real, _schechter(x_real, alpha=alphaFmcmc, phi=phiFmcmc, Mo=MoFmcmc),
        c='g')
del realfunc, x_real
# print(realfunc)

ax.set_yscale('log')
ax.set_ylim([1e-17, 1.e10])
ax.set_xlim([-30, Mlim_left])

fig.savefig('tmp2.pdf')

