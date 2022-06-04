#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from . import crSharedArr
Mlim_left = -35.

class NoDataError(Exception):
    pass

def f_single_schechter_den(M, alpha, phi, Mo, V):
    from .schechter_luminosity import f_single_schechter

    return f_single_schechter(M, alpha, phi, Mo)*V


def ln_likelihood_1Sch(theta, Mlim_right):
    from scipy.integrate import quad
    from .schechter_luminosity import f_single_schechter
    from . import globMod
    x = globMod.x

    alpha, phi, Mo = theta
    n=x.size

    out = (-quad(f_single_schechter, Mlim_left, Mlim_right, args=(alpha, phi, Mo))[0]
            + np.sum(np.log(f_single_schechter(x, alpha=alpha, phi=phi, Mo=Mo))))

    return out

def ln_likelihood_1Sch_densities0(theta, x, xIDs, IDs, V, Mlim_right):
    """
    Note: IDs is assumed to have uniques integer values
    x.size == xIDs.size
    V.size == IDs.size
    """
    from scipy.integrate import quad

    alpha, phi, Mo = theta
    n=x.size
    M = IDs.size
    out = 0.

    for j, (ID, v) in enumerate(zip(IDs, V)):
        ind = xIDs == ID
        out += (-quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alpha,
            phi, Mo, v))[0] + np.sum(np.log(f_single_schechter_den(x[ind],
                alpha=alpha, phi=phi, Mo=Mo, V=v))))

    return out

def ln_likelihood_2SchfixedMF_densities0(theta, x, xIDs, IDs, V, Mlim_right, DeltaM):
    """
    Note: IDs is assumed to have uniques integer values
    x.size == xIDs.size
    V.size == IDs.size
    """
    from scipy.integrate import quad

    alpha, phi, Mo, alphaF, phiF = theta
    n=x.size
    M = IDs.size
    out = 0.

    for j, (ID, v) in enumerate(zip(IDs, V)):
        ind = xIDs == ID
        out += ( -quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alpha , phi , Mo,v ))[0]
                -quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alphaF, phiF,
                    Mo+DeltaM, v))[0]

                + np.sum(np.log(f_single_schechter_den(x[ind], alpha=alpha , phi=phi
                    , Mo=Mo,V=v )
                + f_single_schechter_den(x[ind], alpha=alphaF, phi=phiF,
                    Mo=Mo+DeltaM, V=v)))
                )

    return out


def ln_likelihood_2SchfixedMF_densities1(theta, x, xIDs, IDs, V, inds, Mlim_right, DeltaM):
    """
    Note: IDs is assumed to have uniques integer values
    x.size == xIDs.size
    V.size == IDs.size
    len(ind) == V.size
    """
    from scipy.integrate import quad

    alpha, phi, Mo, alphaF, phiF = theta
    n=x.size
    M = IDs.size
    out = 0.

    for ind, v in zip(inds, V):
        out += ( -quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alpha , phi , Mo,v ))[0]
                -quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alphaF, phiF,
                    Mo+DeltaM, v))[0]

                + np.sum(np.log(f_single_schechter_den(x[ind], alpha=alpha , phi=phi
                    , Mo=Mo,V=v )
                + f_single_schechter_den(x[ind], alpha=alphaF, phi=phiF,
                    Mo=Mo+DeltaM, V=v)))
                )

    return out

def ln_likelihood_2SchfixedMF_densities2(theta, x, xIDs, IDs, V, dims, offsets, Mlim_right, DeltaM):
    """
    Note: IDs is assumed to have uniques integer values
    x.size == xIDs.size
    V.size == IDs.size
    """
    from scipy.integrate import quad

    alpha, phi, Mo, alphaF, phiF = theta
    n=x.size
    M = IDs.size

    out = [ (-quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alpha , phi , Mo,v ))[0]
                -quad(f_single_schechter_den, Mlim_left, Mlim_right, args=(alphaF, phiF,
                    Mo+DeltaM, v))[0]

                + np.sum(np.log(f_single_schechter_den(x[off:off+dim], alpha=alpha , phi=phi
                    , Mo=Mo,V=v )
                    + f_single_schechter_den(x[off:off+dim], alpha=alphaF, phi=phiF,
                    Mo=Mo+DeltaM, V=v)))) for dim, off, v in zip(dims, offsets,
                        V)]
    return sum(out)

def ln_likelihood_2SchfixedMF_densities(theta, Mlim_right, DeltaM):
    """
    Note: IDs is assumed to have uniques integer values
    x.size == xIDs.size
    V.size == IDs.size
    """
    from scipy.integrate import quad
    from .schechter_luminosity import f_single_schechter
    from . import globMod
    x = globMod.x
    xIDs   = globMod.xIDs
    IDs    = globMod.IDs
    V      = globMod.V
    dims   = globMod.dims
    offsets = globMod.offsets


    alpha, phi, Mo, alphaF, phiF = theta
    n=x.size
    M = IDs.size

    integral = ( -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alpha , phi , Mo))[0]
            -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alphaF, phiF,
                Mo+DeltaM))[0])


    out = [np.sum(np.log(f_single_schechter_den(x[off:off+dim], alpha=alpha , phi=phi
                    , Mo=Mo,V=v )
                    + f_single_schechter_den(x[off:off+dim], alpha=alphaF, phi=phiF,
                    Mo=Mo+DeltaM, V=v))) for dim, off, v in zip(dims, offsets, V)]

    return sum(out) + integral * np.sum(V)

def ln_likelihood_1Sch_densities(theta, Mlim_right):
    """
    Note: IDs is assumed to have uniques integer values
    x.size == xIDs.size
    V.size == IDs.size
    """
    from scipy.integrate import quad
    from .schechter_luminosity import f_single_schechter
    from . import globMod

    x = globMod.x
    xIDs = globMod.xIDs
    IDs = globMod.IDs
    V = globMod.V
    dims = globMod.dims
    offsets = globMod.offsets

    alpha, phi, Mo = theta
    n=x.size
    M = IDs.size
    out = 0.

    integral = -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alpha,
        phi, Mo))[0]


    out = [np.sum(np.log(f_single_schechter_den(x[off:off+dim],
            alpha=alpha, phi=phi, Mo=Mo, V=v))) for dim, off, v in zip(dims, offsets, V)]

    return sum(out) + integral * np.sum(V)


def ln_prior_1Sch(theta):

    alpha, phi, Mo = theta

    if -2.5 < alpha < 4. and -23. < Mo < -19.5 and 0. < phi < 1.e5:
        return 0.

    return -np.inf

def ln_prior_1Sch_densities(theta):

    alpha, phi, Mo = theta

    if -2.5 < alpha < 4. and -26. < Mo < -18.0 and 0. < phi < 1.e5:
        return 0.

    return -np.inf


def ln_probability_1Sch(theta, Mlim_right):
    lp = ln_prior_1Sch(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_1Sch(theta, Mlim_right)

def ln_probability_1Sch_densities(theta, Mlim_right):
    lp = ln_prior_1Sch_densities(theta)
    if not np.isfinite(lp):
        return -np.inf
    ln_lk = ln_likelihood_1Sch_densities(theta, Mlim_right) 
    if not np.isfinite(ln_lk):
        raise ValueError('Non finite likelihood with theta = ', theta,
                ' try using data as float64')
    return lp + ln_lk


def ln_probability_2SchfixedMF_densities(theta, Mlim_right, DeltaM):
    lp = ln_prior_2SchfixedMF_densities(theta)
    if not np.isfinite(lp):
        return -np.inf
    ln_lk = ln_likelihood_2SchfixedMF_densities(theta, Mlim_right, DeltaM)
    if not np.isfinite(ln_lk):
        raise ValueError('Non finite likelihood with theta = ', theta,
                ' try using data as float64')
    return lp + ln_lk


def ln_likelihood_2Sch(theta, Mlim_right):
    from scipy.integrate import quad
    from .schechter_luminosity import f_single_schechter
    from . import globMod

    x = globMod.x

    alpha, phi, Mo, alphaF, phiF, MoF = theta
    n=x.size

    out = ( -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alpha , phi , Mo ))[0]
            -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alphaF, phiF, MoF))[0]

            + np.sum(np.log(f_single_schechter(x, alpha=alpha , phi=phi , Mo=Mo )
            + f_single_schechter(x, alpha=alphaF, phi=phiF, Mo=MoF)))
            )

    return out

def ln_likelihood_2SchfixedMF(theta, Mlim_right, DeltaM):
    from scipy.integrate import quad
    from .schechter_luminosity import f_single_schechter
    from . import globMod

    x = globMod.x

    alpha, phi, Mo, alphaF, phiF = theta
    n=x.size

    out = ( -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alpha , phi , Mo ))[0]
            -quad(f_single_schechter, Mlim_left, Mlim_right, args=(alphaF, phiF,
                Mo+DeltaM))[0]

            + np.sum(np.log(f_single_schechter(x, alpha=alpha , phi=phi , Mo=Mo )
            + f_single_schechter(x, alpha=alphaF, phi=phiF, Mo=Mo+DeltaM)))
            )

    return out



def ln_prior_2Sch(theta):

    alpha, phi, Mo, alphaF, phiF, MoF = theta

    if -2.5 < alpha < 20. and -23. < Mo < -19.5 and 0. < phi < 1.e5 \
            and -14.5 > MoF > Mo + .5 and -2.5 < alphaF < alpha  and  \
            0. < phiF < 1.e6:
        return 0.

    return -np.inf

def ln_prior_2SchfixedMF_densities(theta):

    alpha, phi, Mo, alphaF, phiF = theta

    if -2.5 < alpha < 20. and -26. < Mo < -18.0 and 0. < phi < 1.e5 \
            and -2.5 < alphaF < alpha  and  0. < phiF < phi:
        return 0.

    return -np.inf

def ln_prior_2SchfixedMF(theta):

    alpha, phi, Mo, alphaF, phiF = theta

    if -2.5 < alpha < 20. and -23. < Mo < -19.5 and 0. < phi < 1.e5 \
            and -2.5 < alphaF < alpha  and  0. < phiF < phi:
        return 0.

    return -np.inf

def ln_probability_2Sch(theta, Mlim_right):
    lp = ln_prior_2Sch(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_2Sch(theta, Mlim_right)

def ln_probability_2SchfixedMF(theta, Mlim_right, DeltaM):
    lp = ln_prior_2SchfixedMF(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood_2SchfixedMF(theta, Mlim_right, DeltaM)


def checkSort(IDs_in, V_in):
    IDs, index = np.unique(IDs_in, return_index=True)
    V = np.copy(V_in[index])

    if IDs.size != IDs_in.size:
        raise ValueError('The cluster ID array contains repeated values')

    return IDs, V

def mcmc_1Sch_fit(x, nThreads = 8, alpha = -1.4, phi = 38000., Mo = -21.78,
        Mlim_right = -15, nwalkers = 30, nsteps = 5000, saveChain = False,
        chainH5File = None, groupName = 'mcmc', nburninDiscard = 1000):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"
    globMod.x = crSharedArr(x)

    if saveChain and chainH5File is None:
        raise IOError('chainH5File must be a valid hdf5 file handler')

    initial = np.array([alpha, phi, Mo])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_1Sch, args=(Mlim_right,))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_1Sch, args=(Mlim_right,), pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)
    # alphamcmc= np.median(flat_samples[:,0])
    # phimcmc= np.median(flat_samples[:,1])
    # Momcmc= np.median(flat_samples[:,2])

    alpha16, alpha50, alpha84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    phi16, phi50, phi84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    Mo16, Mo50, Mo84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('Mlim_right', Mlim_right)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('alpha_in', alpha)
        group.attrs.create('phi_in', phi)
        group.attrs.create('Mo_in', Mo)


        group2 = chainH5File.create_group(groupName+'/results')
        likelihood_medians = ln_likelihood_1Sch([alpha50, phi50, Mo50], Mlim_right)

        group2.attrs.create('likelihood_medians', likelihood_medians)

        dset = chainH5File.create_dataset(groupName+'/samples', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alpha',
                data=np.array([alpha50-alpha16, alpha50, alpha84-alpha50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phi',
                data=np.array([phi50-phi16, phi50, phi84-phi50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/Mo',
                data=np.array([Mo50-Mo16, Mo50, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/Mr', data=
                np.asarray(x), compression="gzip", compression_opts=9)


    return (sampler, likelihood_medians, [alpha50-alpha16, alpha50, alpha84-alpha50],
            [phi50-phi16, phi50, phi84-phi50],
            [Mo50-Mo16, Mo50, Mo84-Mo50])

def mcmc_1Sch_fit_densities(x_in, xIDs_in, IDs_in, V_in, nThreads = 8, alpha = -1.4, 
        phi = 38000., Mo = -21.78,
        Mlim_right = -15, nwalkers = 30, nsteps = 5000, saveChain = False,
        chainH5File = None, groupName = 'mcmc', nburninDiscard = 1000):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"

    xIDs = np.asarray(np.copy(xIDs_in))
    IDs_in  = np.asarray(IDs_in)
    V_in    = np.asarray(V_in)

    

    if x_in.size != xIDs.size:
        raise ValueError('x_in.size must == xIDs.size')

    if IDs_in.size != V_in.size:
        raise ValueError('V.size must == IDs_in.size')

    if saveChain and chainH5File is None:
        raise IOError('chainH5File must be a valid hdf5 file handler')

    if xIDs.size == 0:
        raise NoDataError

    IDs, V = checkSort(IDs_in, V_in)

    initial = np.array([alpha, phi, Mo])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)
    pos[:,1] = np.abs(pos[:,1])

    # actually it's already ordered
    ind = np.argsort(xIDs)
    uniqueSorted, dims = np.unique(xIDs, return_counts=True)
    offset = np.concatenate( ([0], np.cumsum(dims[:-1])) )
    
    # print(dims, dims.size)
    # print(V.size)
    ind2 = np.isin(IDs, uniqueSorted)
    IDs = IDs[ind2]
    V = V[ind2]
    assert V.size == IDs.size
    assert dims.size == IDs.size
    assert offset.size == IDs.size
    
    x=x_in[ind]
    xIDs = xIDs[ind]

    assert np.array_equal(IDs, uniqueSorted)

    globMod.x   = crSharedArr(x)
    globMod.xIDs= crSharedArr(xIDs)
    globMod.IDs = crSharedArr(IDs)
    globMod.V   = crSharedArr(V)
    globMod.dims =crSharedArr( dims)
    globMod.offsets = crSharedArr(offset)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_1Sch_densities, args=(Mlim_right,))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_1Sch_densities, args=(Mlim_right,), pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)
    emcee.autocorr.integrated_time(sampler.get_chain(discard=nburninDiscard),
            quiet=True)
    # alphamcmc= np.median(flat_samples[:,0])
    # phimcmc= np.median(flat_samples[:,1])
    # Momcmc= np.median(flat_samples[:,2])

    alpha16, alpha50, alpha84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    phi16, phi50, phi84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    Mo16, Mo50, Mo84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('Mlim_right', Mlim_right)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('alpha_in', alpha)
        group.attrs.create('phi_in', phi)
        group.attrs.create('Mo_in', Mo)


        group2 = chainH5File.create_group(groupName+'/results')
        likelihood_medians = ln_likelihood_1Sch([alpha50, phi50, Mo50], Mlim_right)

        group2.attrs.create('likelihood_medians', likelihood_medians)

        dset = chainH5File.create_dataset(groupName+'/samples', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alpha',
                data=np.array([alpha50-alpha16, alpha50, alpha84-alpha50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phi',
                data=np.array([phi50-phi16, phi50, phi84-phi50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/Mo',
                data=np.array([Mo50-Mo16, Mo50, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/Mr', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/IDs', data=
                np.asarray(xIDs), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/ID_unique', data=
                np.asarray(IDs), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/V', data=
                np.asarray(V), compression="gzip", compression_opts=9)


    return (sampler, likelihood_medians, [alpha50-alpha16, alpha50, alpha84-alpha50],
            [phi50-phi16, phi50, phi84-phi50],
            [Mo50-Mo16, Mo50, Mo84-Mo50], ind)

def mcmc_2Sch_fit(x, nThreads = 8, alpha = -1.4, phi = 38000., Mo = -21.78,
        alphaF = -1.5, phiF = 38000., MoF = -24.,
        Mlim_right = -15, nwalkers = 30, nsteps = 5000, saveChain = False,
        chainH5File = None, groupName = 'mcmc', nburninDiscard = 1000):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"

    if saveChain and chainH5File is None:
        raise IOError('chainH5File must be a valid hdf5 file handler')

    if x.size == 0:
        raise NoDataError

    initial = np.array([alpha, phi, Mo, alphaF, phiF, MoF])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)

    globMod.x = crSharedArr(x)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_2Sch, args=(Mlim_right,))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_2Sch, args=(Mlim_right,), pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)
    # alphamcmc= np.median(flat_samples[:,0])
    # phimcmc= np.median(flat_samples[:,1])
    # Momcmc= np.median(flat_samples[:,2])

    alpha16, alpha50, alpha84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    phi16, phi50, phi84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    Mo16, Mo50, Mo84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    alphaF16, alphaF50, alphaF84 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])
    phiF16, phiF50, phiF84 = np.quantile(flat_samples[:,4], [0.16, 0.5, .84])
    MoF16, MoF50, MoF84 = np.quantile(flat_samples[:,5], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('Mlim_right', Mlim_right)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('alpha_in', alpha)
        group.attrs.create('phi_in', phi)
        group.attrs.create('Mo_in', Mo)
        group.attrs.create('alphaF_in', alphaF)
        group.attrs.create('phiF_in', phiF)
        group.attrs.create('MoF_in', MoF)


        group2 = chainH5File.create_group(groupName+'/results')
        likelihood_medians = ln_likelihood_2Sch(
                [alpha50, phi50, Mo50, alphaF50, phiF50, MoF50], Mlim_right)

        group2.attrs.create('likelihood_medians', likelihood_medians)

        dset = chainH5File.create_dataset(groupName+'/samples', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alpha',
                data=np.array([alpha50-alpha16, alpha50, alpha84-alpha50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phi',
                data=np.array([phi50-phi16, phi50, phi84-phi50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/Mo',
                data=np.array([Mo50-Mo16, Mo50, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alphaF',
                data=np.array([alphaF50-alphaF16, alphaF50, alphaF84-alphaF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phiF',
                data=np.array([phiF50-phiF16, phiF50, phiF84-phiF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/MoF',
                data=np.array([MoF50-MoF16, MoF50, MoF84-MoF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/Mr', data=
                np.asarray(x), compression="gzip", compression_opts=9)


    return (sampler, likelihood_medians, [alpha50-alpha16, alpha50, alpha84-alpha50],
            [phi50-phi16, phi50, phi84-phi50],
            [Mo50-Mo16, Mo50, Mo84-Mo50],
            [alphaF50-alphaF16, alphaF50, alphaF84-alphaF50],
            [phiF50-phiF16, phiF50, phiF84-phiF50],
            [MoF50-MoF16, MoF50, MoF84-MoF50])

def mcmc_2Sch_fit_fixedMF(x, nThreads = 8, alpha = -1.4, phi = 38000., Mo = -21.78,
        alphaF = -1.5, phiF = 38000., DeltaM = 0.,
        Mlim_right = -15, nwalkers = 30, nsteps = 5000, saveChain = False,
        chainH5File = None, groupName = 'mcmc', nburninDiscard = 1000):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"

    if saveChain and chainH5File is None:
        raise IOError('chainH5File must be a valid hdf5 file handler')

    initial = np.array([alpha, phi, Mo, alphaF, phiF])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)

    globMod.x = crSharedArr(x)


    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_2SchfixedMF, args=(Mlim_right, DeltaM))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_2SchfixedMF, args=(Mlim_right, DeltaM), 
                    pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)
    # alphamcmc= np.median(flat_samples[:,0])
    # phimcmc= np.median(flat_samples[:,1])
    # Momcmc= np.median(flat_samples[:,2])

    alpha16, alpha50, alpha84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    phi16, phi50, phi84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    Mo16, Mo50, Mo84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    alphaF16, alphaF50, alphaF84 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])
    phiF16, phiF50, phiF84 = np.quantile(flat_samples[:,4], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('Mlim_right', Mlim_right)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('alpha_in', alpha)
        group.attrs.create('phi_in', phi)
        group.attrs.create('Mo_in', Mo)
        group.attrs.create('alphaF_in', alphaF)
        group.attrs.create('phiF_in', phiF)
        group.attrs.create('MoF_in', Mo+DeltaM)
        group.attrs.create('DeltaM', DeltaM)


        group2 = chainH5File.create_group(groupName+'/results')
        likelihood_medians = ln_likelihood_2Sch(
                [alpha50, phi50, Mo50, alphaF50, phiF50, Mo50+DeltaM], Mlim_right)

        group2.attrs.create('likelihood_medians', likelihood_medians)

        dset = chainH5File.create_dataset(groupName+'/samples', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alpha',
                data=np.array([alpha50-alpha16, alpha50, alpha84-alpha50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phi',
                data=np.array([phi50-phi16, phi50, phi84-phi50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/Mo',
                data=np.array([Mo50-Mo16, Mo50, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alphaF',
                data=np.array([alphaF50-alphaF16, alphaF50, alphaF84-alphaF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phiF',
                data=np.array([phiF50-phiF16, phiF50, phiF84-phiF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/MoF',
                data=np.array([Mo50-Mo16, Mo50+DeltaM, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/Mr', data=
                np.asarray(x), compression="gzip", compression_opts=9)


    return (sampler, likelihood_medians, [alpha50-alpha16, alpha50, alpha84-alpha50],
            [phi50-phi16, phi50, phi84-phi50],
            [Mo50-Mo16, Mo50, Mo84-Mo50],
            [alphaF50-alphaF16, alphaF50, alphaF84-alphaF50],
            [phiF50-phiF16, phiF50, phiF84-phiF50])

def mcmc_2Sch_fit_fixedMF_densities(x_in,xIDs_in, IDs_in, V_in, nThreads = 8, 
        alpha = -1.4, phi = 38000., Mo = -21.78,
        alphaF = -1.5, phiF = 38000., DeltaM = 0.,
        Mlim_right = -15, nwalkers = 30, nsteps = 5000, saveChain = False,
        chainH5File = None, groupName = 'mcmc', nburninDiscard = 1000):

    import emcee
    from multiprocessing import Pool
    import os
    from . import globMod
    os.environ["OMP_NUM_THREADS"] = "1"

    xIDs = np.asarray(np.copy(xIDs_in))
    IDs_in  = np.asarray(IDs_in)
    V_in    = np.asarray(V_in)


    if xIDs.size == 0:
        raise NoDataError

    if x_in.size != xIDs.size:
        raise ValueError('x_in.size must == xIDs.size')

    if IDs_in.size != V_in.size:
        raise ValueError('V.size must == IDs_in.size')

    if saveChain and chainH5File is None:
        raise IOError('chainH5File must be a valid hdf5 file handler')

    IDs, V = checkSort(IDs_in, V_in)

    initial = np.array([alpha, phi, Mo, alphaF, phiF])
    ndim = initial.size
    pos = initial + 1e-3*np.random.randn(nwalkers, ndim)
    pos[:,1]=np.abs(pos[:,1])
    pos[:,4]=np.abs(pos[:,4])


    # inds = [xIDs == ID for ID in IDs]

    # assert len(inds) == V.size

    # actually it's already ordered
    ind = np.argsort(xIDs)
    uniqueSorted, dims = np.unique(xIDs, return_counts=True)
    offset = np.concatenate( ([0], np.cumsum(dims[:-1])) )
    
    # print(dims, dims.size)
    # print(V.size)
    ind2 = np.isin(IDs, uniqueSorted)
    IDs = IDs[ind2]
    V = V[ind2]
    assert V.size == IDs.size
    assert dims.size == IDs.size
    assert offset.size == IDs.size
    
    x=x_in[ind]
    xIDs = xIDs[ind]

    assert np.array_equal(IDs, uniqueSorted)

    globMod.x      = crSharedArr(x)
    globMod.xIDs   = crSharedArr(xIDs)
    globMod.IDs    = crSharedArr(IDs)
    globMod.V      = crSharedArr(V)
    globMod.dims   = crSharedArr(dims)
    globMod.offsets = crSharedArr(offset)



    if nThreads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                ln_probability_2SchfixedMF_densities, args=(Mlim_right, DeltaM))
        sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        with Pool(processes = nThreads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                    ln_probability_2SchfixedMF_densities, args=(Mlim_right, DeltaM), 
                    pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=nburninDiscard, thin=1, flat=True)
    emcee.autocorr.integrated_time(sampler.get_chain(discard=nburninDiscard),
            quiet=True)
    # alphamcmc= np.median(flat_samples[:,0])
    # phimcmc= np.median(flat_samples[:,1])
    # Momcmc= np.median(flat_samples[:,2])

    alpha16, alpha50, alpha84 = np.quantile(flat_samples[:,0], [0.16, 0.5, .84])
    phi16, phi50, phi84 = np.quantile(flat_samples[:,1], [0.16, 0.5, .84])
    Mo16, Mo50, Mo84 = np.quantile(flat_samples[:,2], [0.16, 0.5, .84])

    alphaF16, alphaF50, alphaF84 = np.quantile(flat_samples[:,3], [0.16, 0.5, .84])
    phiF16, phiF50, phiF84 = np.quantile(flat_samples[:,4], [0.16, 0.5, .84])

    if saveChain:

        samples = sampler.get_chain()

        group = chainH5File.create_group(groupName)
        group.attrs.create('nwalkers', nwalkers)
        group.attrs.create('ndim', ndim)
        group.attrs.create('Mlim_right', Mlim_right)
        group.attrs.create('nburninDiscard', nburninDiscard)
        group.attrs.create('nsteps', nsteps)
        group.attrs.create('nThreads', nThreads)
        group.attrs.create('alpha_in', alpha)
        group.attrs.create('phi_in', phi)
        group.attrs.create('Mo_in', Mo)
        group.attrs.create('alphaF_in', alphaF)
        group.attrs.create('phiF_in', phiF)
        group.attrs.create('MoF_in', Mo+DeltaM)
        group.attrs.create('DeltaM', DeltaM)


        group2 = chainH5File.create_group(groupName+'/results')
        likelihood_medians = ln_likelihood_2Sch(
                [alpha50, phi50, Mo50, alphaF50, phiF50, Mo50+DeltaM], Mlim_right)

        group2.attrs.create('likelihood_medians', likelihood_medians)

        dset = chainH5File.create_dataset(groupName+'/samples', data=
                samples, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/pos', data=
                pos, compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alpha',
                data=np.array([alpha50-alpha16, alpha50, alpha84-alpha50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phi',
                data=np.array([phi50-phi16, phi50, phi84-phi50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/Mo',
                data=np.array([Mo50-Mo16, Mo50, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/alphaF',
                data=np.array([alphaF50-alphaF16, alphaF50, alphaF84-alphaF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/phiF',
                data=np.array([phiF50-phiF16, phiF50, phiF84-phiF50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/results/MoF',
                data=np.array([Mo50-Mo16, Mo50+DeltaM, Mo84-Mo50]),
                compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/Mr', data=
                np.asarray(x), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/IDs', data=
                np.asarray(xIDs), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/ID_unique', data=
                np.asarray(IDs), compression="gzip", compression_opts=9)

        dset = chainH5File.create_dataset(groupName+'/data_sample/V', data=
                np.asarray(V), compression="gzip", compression_opts=9)


    return (sampler, likelihood_medians, [alpha50-alpha16, alpha50, alpha84-alpha50],
            [phi50-phi16, phi50, phi84-phi50],
            [Mo50-Mo16, Mo50, Mo84-Mo50],
            [alphaF50-alphaF16, alphaF50, alphaF84-alphaF50],
            [phiF50-phiF16, phiF50, phiF84-phiF50], ind)
