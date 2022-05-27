#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from . import crSharedArr

def ln_likelihood(theta, x, y, yerr):
    a, b, c = theta
    model = a*x**2 + b*x + c
    sigma2 = yerr**2 #+ model**2
    out = -0.5*np.sum( np.log(2*np.pi*sigma2) + (y-model)**2/sigma2)
    return out

def ln_likelihood_linear(theta, x, y, yerr):
    m, q = theta
    model = m*x+q
    sigma2 = yerr**2 #+ model**2
    out = -0.5*np.sum( np.log(2*np.pi*sigma2) + (y-model)**2/sigma2)
    return out

def ln_likelihood_linearSigma(theta):
    from . import globMod
    m, q, s2 = theta
    model = m*globMod.x+q

    # here the sigma is given by the uncerts yerr + the sigma we are fitting
    sigma2 = globMod.yerr**2 + s2   
    out = -0.5*np.sum( np.log(2*np.pi*sigma2) + (globMod.y-model)**2/sigma2)
    return out

def ln_likelihood_2PowerLawsSigma(theta):
    from . import globMod
    m1, q1, s2, m2, x0 = theta
    # model = m*globMod.x+q

    mask = globMod.x <= x0
    mask2 = np.logical_not(mask)
    model = np.empty_like(globMod.x)
    model[mask] = m1*globMod.x[mask] + q1
    model[mask2] = m2*globMod.x[mask2] + (m1-m2)*x0 + q1

    # here the sigma is given by the uncerts yerr + the sigma we are fitting
    sigma2 = globMod.yerr**2 + s2
    out = -0.5*np.sum( np.log(2*np.pi*sigma2) + (globMod.y-model)**2/sigma2)
    return out

def ln_likelihood_linear2Gauss(theta, x, y, yerr):
    m, q, s2, mu_2, s2_2 = theta
    errors = y - (m*x+q)

    # here the sigma of the first gaussian is given by the uncerts yerr + the sigma 
    # we are fitting
    sigma2 = yerr**2 + s2
    out = np.sum(-np.log(2.) + np.log(
        np.exp(-0.5*errors**2/sigma2)         /np.sqrt(2.*np.pi*sigma2)   +
        np.exp(-0.5*(errors-mu_2)**2/s2_2)/np.sqrt(2.*np.pi*s2_2)   
        ))
    return out


def ln_likelihood_2PowerLaws2Gauss(theta):
    from . import globMod
    m1, q1, s2, mu_2, s2_2, m2, x0 = theta

    mask = globMod.x <= x0
    mask2 = np.logical_not(mask)
    model = np.empty_like(globMod.x)
    model[mask] = m1*globMod.x[mask] + q1
    model[mask2] = m2*globMod.x[mask2] + (m1-m2)*x0 + q1

    errors = globMod.y - model

    # here the sigma of the first gaussian is given by the uncerts yerr + the sigma 
    # we are fitting
    sigma2 = globMod.yerr**2 + s2
    out = np.sum(-np.log(2.) + np.log(
        np.exp(-0.5*errors**2/sigma2)         /np.sqrt(2.*np.pi*sigma2)   +
        np.exp(-0.5*(errors-mu_2)**2/s2_2)/np.sqrt(2.*np.pi*s2_2)   
        ))
    return out

def halfGauss(errors, s2_2):
    out = np.zeros_like(errors)
    mask = errors <= 0.
    mu_2 = 0.
    out[mask] = np.exp(-0.5*((errors[mask]-mu_2)/s2_2)**2)/np.sqrt(2.*np.pi*s2_2)
    return out

def ln_likelihood_linear1p5Gauss(theta):
    from . import globMod
    m, q, s2, s2_2 = theta
    errors = globMod.y - (m*globMod.x+q)

    # here the sigma of the first gaussian is given by the uncerts yerr + the sigma 
    # we are fitting
    sigma2 = globMod.yerr**2 + s2
    out = np.sum(np.log(1.5) + np.log(
        np.exp(-0.5*(errors/sigma2)**2)         /np.sqrt(2.*np.pi*sigma2)   +
        halfGauss(errors, s2_2)
        ))
    return out

def ln_likelihood_linear_log(theta, x, y, yerr):
    m, q = theta
    model = 10.**(m*x+q)
    sigma2 = yerr**2 #+ model**2
    out = -0.5*np.sum(np.log(2*np.pi*sigma2) + (y-model)**2/sigma2)
    return out


def ln_prior_linear(theta):
    m, q = theta
    if -1000. < q < 1000.:
        return 0.0
    return -np.inf

def ln_prior_linearSigma(theta):
    m, q, sigma2 = theta
    if -1000. < q < 1000. and sigma2 > 0.:
        return 0.0
    return -np.inf

def ln_prior_2PowerLawsSigma(theta):
    m1, q1, sigma2, m2, x0 = theta
    if -1000. < q1 < 1000. and sigma2 > 0. and 9.8<x0<11. and \
            -0.5<np.arctan(m2)< np.pi/2. and -0.5<np.arctan(m1)< np.pi/2.:
                return 0.0
    return -np.inf

def ln_prior_linear2Gauss(theta):
    m, q, sigma2, mu_2, sigma2_2 = theta
    if -1000. < q < 1000. and sigma2 > 0. and -10.<mu_2 < 10. and sigma2_2 > sigma2:
        return 0.0
    return -np.inf

def ln_prior_2PowerLaws2Gauss(theta):
    m1, q1, sigma2, mu_2, sigma2_2, m2, x0 = theta
    if -1000. < q1 < 1000. and sigma2 > 0. and -10.<mu_2 < 10. and \
            sigma2_2 > sigma2 and -0.5<np.arctan(m2)< np.pi/2. and \
            0.0<np.arctan(m1)< np.pi/2. and 10.<x0<11. and m2<=m1:
                return 0.0
    return -np.inf

def ln_prior_linear1p5Gauss(theta):
    m, q, sigma2, sigma2_2 = theta
    # if -1000. < q < 1000. and sigma2 > 0. and sigma2 < 0.7 and sigma2_2 > 0.:
    if 5.17 < q < 5.5 and m< -2.34 and sigma2 > 0. and sigma2 < 0.7 and sigma2_2 > 0.:
        return 0.0
    return -np.inf

def ln_prior(theta):
    a, b, c = theta
    # if 0. < a < 100.  and b < 0.:
    # if 0. < a < 100.:
    if 0. < np.abs(a) < 100.:
        return 0.0
    return -np.inf


def ln_probability(theta, x, y, yerr):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta, x, y, yerr)


def ln_probability_linear(theta, x, y, yerr):
    lp = ln_prior_linear(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_linear(theta, x, y, yerr)

def ln_probability_linearSigma(theta):
    lp = ln_prior_linearSigma(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_linearSigma(theta)

def ln_probability_2PowerLawsSigma(theta):
    lp = ln_prior_2PowerLawsSigma(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_2PowerLawsSigma(theta)

def ln_probability_linear2Gauss(theta, x, y, yerr):
    lp = ln_prior_linear2Gauss(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_linear2Gauss(theta, x, y, yerr)

def ln_probability_2PowerLaws2Gauss(theta):
    lp = ln_prior_2PowerLaws2Gauss(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_2PowerLaws2Gauss(theta)

def ln_probability_linear1p5Gauss(theta):
    lp = ln_prior_linear1p5Gauss(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_linear1p5Gauss(theta)

def ln_probability_linear_log(theta, x, y, yerr):
    lp = ln_prior_linear(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_linear_log(theta, x, y, yerr)


def mcmc_fit(x, y, yerr, nThreads = 8, nburninDiscard = 1000, nwalkers =
        8, nsteps = 5000, a=1.,b=-1.,c=-1., saveChain = False, 
        chainH5File = None, groupName = 'mcmc'):

    import emcee
    from multiprocessing import Pool
    from contextlib import closing
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    initial = np.array([a, b, c])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability, args=(x, y, yerr))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with closing(Pool(processes = nThreads)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability, args=(x, y, yerr), pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
            pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    a16, a50, a84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    b16, b50, b84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    c16, c50, c84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('a_in', a)
        group.attrs.create('b_in', b)
        group.attrs.create('c_in', c)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/a',
                data=np.array([a50-a16, a50, a84-a50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/b',
                data=np.array([b50-b16, b50, b84-b50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/c',
                data=np.array([c50-c16, c50, c84-c50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)


    return (flat_samples , [a50-a16, a50, a84-a50],
            [b50-b16, b50, b84-b50],
            [c50-c16, c50, c84-c50])


def mcmc_fit_linear(x, y, yerr, nThreads = 8, nburninDiscard = 1000, nwalkers =
        8, nsteps = 5000, m=1., q=0., saveChain = False, 
        chainH5File = None, groupName = 'mcmc', log=False):

    import emcee
    from multiprocessing import Pool
    from contextlib import closing
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    initial = np.array([m,q])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)


    if log:
        if nThreads == 1:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_linear_log, args=(x, y, yerr))
            sampler.run_mcmc(pos, nsteps, progress=True)
        else:
            with closing(Pool(processes = nThreads)) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim,
                        ln_probability_linear_log, args=(x, y, yerr), pool=pool)
                sampler.run_mcmc(pos, nsteps, progress=True)
                pool.terminate()
    else:
        if nThreads == 1:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_linear, args=(x, y, yerr))
            sampler.run_mcmc(pos, nsteps, progress=True)
        else:
            with closing(Pool(processes = nThreads)) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim,
                        ln_probability_linear, args=(x, y, yerr), pool=pool)
                sampler.run_mcmc(pos, nsteps, progress=True)
                pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    m16, m50, m84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    q16, q50, q84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('m_in', m)
        group.attrs.create('q_in', q)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m',
                data=np.array([m50-m16, m50, m84-m50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/q',
                data=np.array([q50-q16, q50, q84-q50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)

    return (flat_samples , [m50-m16, m50, m84-m50],
            [q50-q16, q50, q84-q50])

def mcmc_fit_linearSigma(x, y, yerr, nThreads = 8, 
        nburninDiscard = 1000, nwalkers = 8, nsteps = 5000, 
        m=1., q=0., sigma2 = 1., saveChain = False, 
        chainH5File = None, groupName = 'mcmc'):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"

    initial = np.array([m,q, sigma2])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)

    globMod.x = crSharedArr(x)
    globMod.y = crSharedArr(y)
    globMod.yerr = crSharedArr(yerr)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_linearSigma)
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_linearSigma, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
            pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    m16, m50, m84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    q16, q50, q84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    s16, s50, s84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('m_in', m)
        group.attrs.create('q_in', q)
        group.attrs.create('sigma_in', sigma2)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m',
                data=np.array([m50-m16, m50, m84-m50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/q',
                data=np.array([q50-q16, q50, q84-q50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2',
                data=np.array([s50-s16, s50, s84-s50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)

    return (flat_samples , [m50-m16, m50, m84-m50],
            [q50-q16, q50, q84-q50], [s50-s16, s50, s84-s50])



def mcmc_fit_2PowerLawsSigma(x, y, yerr, nThreads = 8, 
        nburninDiscard = 1000, nwalkers = 8, nsteps = 5000, 
        m1=1., q1=0., sigma2 = 1., 
        m2=1., x0=0.,
        saveChain = False, chainH5File = None, groupName = 'mcmc'):
    """
    Fit a broken power law:
         m1*x+q1    x<= x0
    y = 
         m2*x+(m1-m2)*x0+q1    x>x0

    x0 is the conjunction point between the two power laws
    """

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"

    initial = np.array([m1,q1, sigma2, m2, x0])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)

    globMod.x = crSharedArr(x)
    globMod.y = crSharedArr(y)
    globMod.yerr = crSharedArr(yerr)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_2PowerLawsSigma)
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_2PowerLawsSigma, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
            pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    m1_16, m1_50, m1_84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    q1_16, q1_50, q1_84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    s_16, s_50, s_84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])
    m2_16, m2_50, m2_84 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])
    x0_16, x0_50, x0_84 = np.quantile(flat_samples[:,4], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('m1_in', m1)
        group.attrs.create('q1_in', q1)
        group.attrs.create('m2_in', m2)
        group.attrs.create('x0_in', x0)
        group.attrs.create('sigma_in', sigma2)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m1',
                data=np.array([m1_50-m1_16, m1_50, m1_84-m1_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/q1',
                data=np.array([q1_50-q1_16, q1_50, q1_84-q1_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2',
                data=np.array([s_50-s_16, s_50, s_84-s_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m2',
                data=np.array([m2_50-m2_16, m2_50, m2_84-m2_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/x0',
                data=np.array([x0_50-x0_16, x0_50, x0_84-x0_50]),
                compression="gzip", compression_opts=9)


        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)

    return (flat_samples , [m1_50-m1_16, m1_50, m1_84-m1_50],
            [q1_50-q1_16, q1_50, q1_84-q1_50], [s_50-s_16, s_50, s_84-s_50],
            [m2_50-m2_16, m2_50, m2_84-m2_50], 
            [x0_50-x0_16, x0_50, x0_84-x0_50])

def mcmc_fit_linear2Gauss(x, y, yerr, nThreads = 8, 
        nburninDiscard = 1000, nwalkers = 8, nsteps = 5000, 
        m=1., q=0., sigma2 = 1., mu_2 = 0., sigma2_2 = 5.,
        saveChain = False, 
        chainH5File = None, groupName = 'mcmc'):

    import emcee
    from multiprocessing import Pool
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    initial = np.array([m,q, sigma2, mu_2, sigma2_2])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_linear2Gauss, args=(x, y, yerr))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_linear2Gauss, args=(x, y, yerr), pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
            pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    m16, m50, m84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    q16, q50, q84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    s16, s50, s84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])
    mu_216, mu_250, mu_284 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])
    s_216, s_250, s_284 = np.quantile(flat_samples[:,4], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('m_in', m)
        group.attrs.create('q_in', q)
        group.attrs.create('sigma_in', sigma2)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m',
                data=np.array([m50-m16, m50, m84-m50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/q',
                data=np.array([q50-q16, q50, q84-q50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2',
                data=np.array([s50-s16, s50, s84-s50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2_2',
                data=np.array([s_250-s_216, s_250, s_284-s_250]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/mu_2',
                data=np.array([mu_250-mu_216, mu_250, mu_284-mu_250]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)

    return (flat_samples , [m50-m16, m50, m84-m50],
            [q50-q16, q50, q84-q50], [s50-s16, s50, s84-s50],
            [mu_250-mu_216, mu_250, mu_284-mu_250],
            [s_250-s_216, s_250, s_284-s_250])

def mcmc_fit_linear1p5Gauss(x, y, yerr, nThreads = 8, 
        nburninDiscard = 1000, nwalkers = 8, nsteps = 5000, 
        m=1., q=0., sigma2 = 1., sigma2_2 = 1.,
        saveChain = False, 
        chainH5File = None, groupName = 'mcmc'):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"
    globMod.x = crSharedArr(x)
    globMod.y = crSharedArr(y)
    globMod.yerr = crSharedArr(yerr)

    initial = np.array([m,q, sigma2, sigma2_2])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_linear1p5Gauss)
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_linear1p5Gauss, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
            pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    m16, m50, m84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    q16, q50, q84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    s16, s50, s84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])
    s_216, s_250, s_284 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('m_in', m)
        group.attrs.create('q_in', q)
        group.attrs.create('sigma_in', sigma2)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m',
                data=np.array([m50-m16, m50, m84-m50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/q',
                data=np.array([q50-q16, q50, q84-q50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2',
                data=np.array([s50-s16, s50, s84-s50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2_2',
                data=np.array([s_250-s_216, s_250, s_284-s_250]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)

    return (flat_samples , [m50-m16, m50, m84-m50],
            [q50-q16, q50, q84-q50], [s50-s16, s50, s84-s50],
            [s_250-s_216, s_250, s_284-s_250])


def mcmc_fit_2PowerLaws2Gauss(x, y, yerr, nThreads = 8, 
        nburninDiscard = 1000, nwalkers = 8, nsteps = 5000, 
        m1=1., q1=0., sigma2 = 1., mu_2 = 0., sigma2_2 = 5.,
        m2 = 1., x0=6.,
        saveChain = False, 
        chainH5File = None, groupName = 'mcmc'):
    """
    Fit a broken power law:
         m1*x+q1    x<= x0
    y = 
         m2*x+(m1-m2)*x0+q1    x>x0

    x0 is the conjunction point between the two power laws
    Use 2 Gaussians for the errors.
    """

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"
    globMod.x = crSharedArr(x)
    globMod.y = crSharedArr(y)
    globMod.yerr = crSharedArr(yerr)


    initial = np.array([m1,q1, sigma2, mu_2, sigma2_2, m2, x0])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_2PowerLaws2Gauss)
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_2PowerLaws2Gauss, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)
            pool.terminate()

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)

    m1_16, m1_50, m1_84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    q1_16, q1_50, q1_84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    s16, s50, s84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])
    mu_216, mu_250, mu_284 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])
    s_216, s_250, s_284 = np.quantile(flat_samples[:,4], [0.16, 0.5, .84])
    m2_16, m2_50, m2_84 = np.quantile(flat_samples[:,5], [0.16, 0.5, .84])
    x0_16, x0_50, x0_84 = np.quantile(flat_samples[:,6], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('m1_in', m1)
        group.attrs.create('m2_in', m2)
        group.attrs.create('q1_in', q1)
        group.attrs.create('x0_in', x0)
        group.attrs.create('sigma_in', sigma2)
        group.attrs.create('sigma2_in', sigma2_2)


        dset = chainH5File.create_dataset(groupName+'/chain', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m1',
                data=np.array([m1_50-m1_16, m1_50, m1_84-m1_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/q1',
                data=np.array([q1_50-q1_16, q1_50, q1_84-q1_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2',
                data=np.array([s50-s16, s50, s84-s50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/sigma2_2',
                data=np.array([s_250-s_216, s_250, s_284-s_250]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/mu_2',
                data=np.array([mu_250-mu_216, mu_250, mu_284-mu_250]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/m2',
                data=np.array([m2_50-m2_16, m2_50, m2_84-m2_50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/x0',
                data=np.array([x0_50-x0_16, x0_50, x0_84-x0_50]),
                compression="gzip", compression_opts=9)


        dset = chainH5File.create_dataset(groupName+'/data_sample/x', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/y', data=
                np.asarray(y), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/yerr', data=
                np.asarray(yerr), compression="gzip", compression_opts=9)

    return (flat_samples , [m1_50-m1_16, m1_50, m1_84-m1_50],
            [q1_50-q1_16, q1_50, q1_84-q1_50], [s50-s16, s50, s84-s50],
            [mu_250-mu_216, mu_250, mu_284-mu_250],
            [s_250-s_216, s_250, s_284-s_250],
            [m2_50-m2_16, m2_50, m2_84-m2_50],
            [x0_50-x0_16, x0_50, x0_84-x0_50])

def xylim_corner_lin(mRa, qRa, axs):
    axs[0].set_xlim(mRa)
    axs[2].set_xlim(mRa)
    axs[2].set_ylim(qRa)
    axs[3].set_xlim(qRa)

def calc_shaded_region_line(chain, x_plot, m,q, calcDataSpace = True, 
        useEntireChain = True):
    """
    Calculate the shaded region for 1 sigma values in the fit by using the
    chains in the mcmc.

    """
    x = np.asarray(x_plot)

    y_max  = np.full_like(x, -np.inf)
    y_min  = np.full_like(x,  np.inf)

    if calcDataSpace:
        ind = np.arange(chain.shape[0])
    else:
        # find all the triplets that stays in 1 sigma
        ind = np.flatnonzero(
                (chain[:,0] > (m[1] - m[0])) &
                (chain[:,0] < (m[1] + m[2])) &
                (chain[:,1] > (q[1] - q[0])) &
                (chain[:,1] < (q[1] + q[2])) )

    if ind.size > 0:
        if useEntireChain:
            choice = ind
        else:
            choice = np.random.choice(ind, min(1000, ind.size))

        if calcDataSpace:
            samples = np.zeros((choice.size, x.size))
            for ii, i in enumerate(choice):
                samples[ii,:] = x*chain[i,0]+ chain[i, 1]

            y_max = np.quantile(samples, 0.16, axis=0)
            y_min = np.quantile(samples, 0.84, axis=0)
            median1 = np.quantile(samples, 0.5, axis=0)
            assert y_max.size == x.size
            assert y_min.size == x.size
            
        else:
            for i in choice:
                y_max = np.maximum(y_max, x*chain[i,0]+ chain[i, 1])
                y_min = np.minimum(y_min, x*chain[i,0]+ chain[i, 1])

    return y_max, y_min

def calc_shaded_region_2PowerLawsOld(chain, x_plot, useEntireChain = True,
        useSigma = False):
    """
    Calculate the shaded region for 1 sigma values in the fit by using the
    chains in the mcmc.

    """
    x = np.asarray(x_plot)

    y_max  = np.full_like(x, -np.inf)
    y_min  = np.full_like(x,  np.inf)

    ind = np.arange(chain.shape[0])

    if ind.size > 0:
        if useEntireChain:
            choice = ind
        else:
            choice = np.random.choice(ind, min(1000, ind.size))

        samples = np.zeros((choice.size, x.size))
        for ii, i in enumerate(choice):
            mask = x<= chain[i,6]
            mask2 = np.logical_not(mask)
            samples[ii,mask] = x[mask]*chain[i,0]+ chain[i, 1]
            samples[ii,mask2] = x[mask2]*chain[i,5]+ (chain[i,0] - 
                    chain[i,5]) * chain[i,6] + chain[i, 1]
            if useSigma:
                samples[ii,:] += np.random.default_rng().normal(0., 
                        np.sqrt(chain[i, 2]))

        y_max = np.quantile(samples, 0.16, axis=0)
        y_min = np.quantile(samples, 0.84, axis=0)
        median1 = np.quantile(samples, 0.5, axis=0)
        assert y_max.size == x.size
        assert y_min.size == x.size
            
    return y_max, y_min

def calc_shaded_region_2PowerLaws(chain, x_plot):
    """
    Calculate the shaded region for 1 sigma values in the fit by using the
    chains in the mcmc.

    """
    xcp = cp.array(x_plot)
    chaincp = cp.array(chain)

    if chain.shape[0] > 0:

        c0 = cp.tile(chaincp[:,0],[xcp.size,1]).T
        c1 = cp.tile(chaincp[:,1],[xcp.size,1]).T
        c2 = cp.tile(chaincp[:,2],[xcp.size,1]).T
        c3 = cp.tile(chaincp[:,3],[xcp.size,1]).T
        c4 = cp.tile(chaincp[:,4],[xcp.size,1]).T
        c5 = cp.tile(chaincp[:,5],[xcp.size,1]).T
        c6 = cp.tile(chaincp[:,6],[xcp.size,1]).T
        x2 = cp.broadcast_to(xcp,(chain.shape[0],xcp.size))

        mask = x2 <= c6
        mask2 = np.logical_not(mask)

        samples = cp.empty_like(x2)
        samples[mask] = cp.multiply(x2[mask],c0[mask]) + c1[mask]
        samples[mask2] = cp.multiply(x2[mask2],c5[mask2]) + cp.multiply(c0[mask2] - 
                c5[mask2], c6[mask2]) + c1[mask2]

        samples += cupy.random.normal(0., c2, size=samples.shape)

        y_maxcp = cp.quantile(samples, 0.16, axis=0)
        y_mincp = cp.quantile(samples, 0.84, axis=0)
        mediancp = cp.quantile(samples, 0.5, axis=0)
        cp.cuda.Stream.null.synchronize()
            
        y_max  = cp.asnumpy(y_maxcp)
        y_min  = cp.asnumpy(y_mincp)
        median = cp.asnumpy(mediancp)

    return y_max, y_min
