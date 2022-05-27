#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import curve_fit

from mcmc_fits.schechter_luminosity import f_single_schechter as f_single
from mcmc_fits import mcmc_fitP


def compute_luminosity_function(data, min_mag = 0.,
        additional_selection = None, max_mag = -26.5, NumberOfBins = 25):


    nBins = NumberOfBins
    rBins = [max_mag, min_mag]

    iSel_tmp = (data <= min_mag)

    if additional_selection is not None:
        iSel = np.logical_and(iSel_tmp, additional_selection)
    else:
        iSel = iSel_tmp


    unnormedHist, bin_edges = np.histogram(data[iSel], bins=nBins, range=rBins,
            density=False)
    numGalaxies = np.sum(unnormedHist)

    iSel = np.logical_and(iSel, data <= bin_edges[-1])
    iSel = np.logical_and(iSel, data >= bin_edges[0])
    assert numGalaxies == np.count_nonzero(iSel)


    x = (bin_edges[0:bin_edges.size-1] + bin_edges[1:bin_edges.size]) / 2
    dx = x[1] - x[0]
    y = unnormedHist # / (numGalaxies * 1.0) / dx

    xerr = (bin_edges[1:bin_edges.size] - bin_edges[0:bin_edges.size-1]) / 2
    yerr = np.sqrt(unnormedHist) # / numGalaxies / dx

    return x, y, xerr, yerr



def bayesian_fit(data, dataClusterIDs,  IDs, V,  mag_min_fit, nThreads,
        nwalkers, nstepsMCMC, nburninDiscard, chainFile, groupName='mcmc',
        dataUniqueIDs = None, Sc2 = False, Sc1 = False):
    """
    Interface for bayesian fit.

    """

    data = np.asarray(data)
    dataIDs = np.asarray(dataClusterIDs)
    if dataUniqueIDs is not None:
        iSubHaloes = np.asarray(dataUniqueIDs)

    ind = data < mag_min_fit

    data = data[ind]
    dataIDs = dataIDs[ind]
    if dataUniqueIDs is not None:
        iSubHaloes = iSubHaloes[ind]

    # all the dataClusterIDs must be represented in IDs
    tmpUGalID = np.unique(dataClusterIDs)
    tmpUCluID, index = np.unique(IDs, return_index=True)
    V_tmp = V[index]

    if IDs.size != tmpUCluID.size:
        raise ValueError('The cluster ID array contains repeated values')

    mask = np.isin(tmpUCluID, tmpUGalID)
    if not np.array_equal(tmpUGalID, tmpUCluID[mask]):
        raise ValueError('dataClusterIDs must be included in IDs')

    tmpUCluID = tmpUCluID[mask]
    V_tmp = V_tmp[mask]

    mask = np.isin(tmpUGalID, tmpUCluID)
    if not np.array_equal(tmpUGalID[mask], tmpUCluID):
        raise ValueError('The cluster ID array contains values not present'
                ' in the galaxy file')

    np.testing.assert_equal(tmpUGalID, tmpUCluID)
    del tmpUGalID


    try:
        alphatmp, phitmp, Motmp = fit_try(data, mag_min_fit)
        print('fit_try success', alphatmp, phitmp, Motmp)
    except:
        print('fit_try failed')
        alphatmp = -1.4
        phitmp = 0.1
        Motmp = -22.

    if Sc1:
        print('doing 1sc density')
        try:
            if dataIDs.size >0 :
                sampler, lk1, alphamcmc, phimcmc, Momcmc, ind = \
                        mcmc_fitP.mcmc_1Sch_fit_densities(
                        data, dataIDs, tmpUCluID, V_tmp, nThreads = nThreads,
                        Mlim_right=mag_min_fit, saveChain =True,
                        nwalkers  = nwalkers,
                        nsteps = nstepsMCMC,
                        nburninDiscard = nburninDiscard, groupName =
                        groupName+'/1sc_den',
                        chainH5File = chainFile,
                        alpha=alphatmp, phi=phitmp, Mo=Motmp)
                BIC1 = np.log(data.size)*3 - 2.*np.log(lk1)
                chainFile[groupName+'/1sc_den'].attrs.create('BIC', BIC1)
                if dataUniqueIDs is not None:
                    chainFile[groupName+'/1sc_den/data_sample/iSubHaloes'] = iSubHaloes[ind]
            elif dataIDs.size ==0 :
                # save one array to testify that we have an empty dataset
                chainFile.create_dataset(groupName+'/1sc_den/data_sample/IDs', data=
                        dataIDs, compression="gzip", compression_opts=9)
            else:
                raise ValueError


        except ValueError as valerr:
            print('mcmc_1Sch_fit_densities returned a ValueError')
            raise valerr

    if Sc2:
        print('doing 2sc density')
        try:
            if dataIDs.size >0 :
                sampler, lk3, alphamcmc, phimcmc, Momcmc, alphaFmcmc, \
                        phiFmcmc, ind = mcmc_fitP.mcmc_2Sch_fit_fixedMF_densities(
                        data, dataIDs, tmpUCluID, V_tmp, nThreads = nThreads,
                        Mlim_right=mag_min_fit, saveChain =True,
                        nwalkers  = nwalkers,
                        nsteps = nstepsMCMC,
                        nburninDiscard = nburninDiscard, groupName =
                        groupName+'/2scMoF_fixed_den',
                        chainH5File = chainFile,
                        alpha=alphatmp, phi=phitmp, Mo=Motmp,
                        alphaF=alphatmp-0.2, phiF=phitmp, DeltaM=0.)

                BIC3 = np.log(data.size)*5 - 2.*np.log(lk3)
                chainFile[groupName+'/2scMoF_fixed_den'].attrs.create('BIC', BIC3)
                if dataUniqueIDs is not None:
                    chainFile[groupName+'/2scMoF_fixed_den/data_sample'
                            '/iSubHaloes'] = iSubHaloes[ind]

            elif dataIDs.size ==0 :
                # save one array to testify that we have an empty dataset
                chainFile.create_dataset(groupName+'/2scMoF_fixed_den/data_sample/IDs', data=
                        dataIDs, compression="gzip", compression_opts=9)
            else:
                raise ValueError


        except ValueError as valerr:
            print('mcmc_2Sch_fit returned a ValueError')
            raise valerr


def fit_try(data, min_mag):
    """
    A simple non-normalized one-function fit to crudely estimate the parameters to feed to
    bayesian fit (especially phi).

    """

    x, y, xerr, yerr = compute_luminosity_function(data,
            min_mag = min_mag)

    # we need to filter the points
    nonzero = (x < min_mag) & (y > 0)
    x = x[nonzero]
    y = y[nonzero]

    try:
        if y.size > 1:
            single, pcov = curve_fit(f_single, x, y, bounds=([-5.0, 0.0, -26.0],
                [3.0, 1e5, -18.0]))
            if single[2] < -23.:
                single[2] = -23.
            return single
        else:
            return np.array([-1.4, 1., -21.])
    except:
            return np.array([-1.4, 1., -21.])
