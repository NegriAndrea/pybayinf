#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import h5py

from .mcmc_fit import (ln_likelihood_1Sch_densities,
        ln_likelihood_2SchfixedMF_densities)
class EmptyDataBin(Exception):
    pass

class dataSetNotFound(Exception):
    pass


def lk_1sc_den(group, chain):
    groupName = group +'/1sc_den'

    if chain[groupName+'/data_sample/IDs'].size == 0:
        raise EmptyDataBin

    Mlim_right = chain[groupName].attrs['Mlim_right']
    nburninDiscard = chain[groupName].attrs['nburninDiscard']
    nsteps = chain[groupName].attrs['nsteps']
    nwalkers = chain[groupName].attrs['nwalkers']
    ndim = chain[groupName].attrs['ndim']

    alpha = chain[groupName+'/results/alpha'][()]
    Mo = chain[groupName+'/results/Mo'][()]
    phi = chain[groupName+'/results/phi'][()]
    Mr = chain[groupName+'/data_sample/Mr'][()]
    xIDs = chain[groupName+'/data_sample/IDs'][()] 
    IDs = chain[groupName+'/data_sample/ID_unique'][()]
    V = chain[groupName+'/data_sample/V'][()]
    chain_arr = chain[groupName+'/samples'][()]

    # same as the mcmc code
    uniqueSorted, dims = np.unique(xIDs, return_counts=True)
    offset = np.concatenate( ([0], np.cumsum(dims[:-1])) )

    # # cut the chain and flatten it
    # chain1sc = np.vstack(chain_arr[nburninDiscard:, :, :])
    # assert chain1sc.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)
    # ind = np.flatnonzero(
            # (chain1sc[:,0] > (alpha[1] - alpha[0])) &
            # (chain1sc[:,0] < (alpha[1] + alpha[2])) &
            # (chain1sc[:,1] > (phi  [1] - phi  [0])) &
            # (chain1sc[:,1] < (phi  [1] + phi  [2])) &
            # (chain1sc[:,2] > (Mo   [1] - Mo   [0])) &
            # (chain1sc[:,2] < (Mo   [1] + Mo   [2])) )


    # if ind.size > 0:
        # print('aaaaa')
        # choice = np.random.choice(ind, min(10, ind.size))
        # ln_lk_arr = np.zeros(choice.size)

        # ln_lk_arr = [ln_likelihood_1Sch_densities((chain1sc[i,0], chain1sc[i,1], 
            # chain1sc[i,2]), Mr,xIDs, IDs, V, dims, offset, Mlim_right) for ii,
            # i in enumerate(choice)]

    # ln_lk_arr = np.asarray(ln_lk_arr )
    theta = np.copy(alpha[1]), np.copy(phi[1]), np.copy(Mo[1])
    ln_lk = ln_likelihood_1Sch_densities(theta, Mr, xIDs, IDs, V, dims,
                offset, Mlim_right)
    # print(ln_lk, ln_lk_arr.max())

    return ln_lk, Mr.size

def lk_2sc_den(group, chain):
    groupName = group +'/2scMoF_fixed_den'
    try:
        if chain[groupName+'/data_sample/IDs'].size == 0:
            raise EmptyDataBin
    except KeyError:
            raise dataSetNotFound
    Mlim_right = chain[groupName].attrs['Mlim_right']
    DeltaM = chain[groupName].attrs['DeltaM']
    alpha = chain[groupName+'/results/alpha'][1]
    alphaF = chain[groupName+'/results/alphaF'][1]
    Mo = chain[groupName+'/results/Mo'][1]
    phi = chain[groupName+'/results/phi'][1]
    phiF = chain[groupName+'/results/phiF'][1]
    Mr = chain[groupName+'/data_sample/Mr'][()]
    xIDs = chain[groupName+'/data_sample/IDs'][()] 
    IDs = chain[groupName+'/data_sample/ID_unique'][()]
    V = chain[groupName+'/data_sample/V'][()]

    # same as the mcmc code
    uniqueSorted, dims = np.unique(xIDs, return_counts=True)
    offset = np.concatenate( ([0], np.cumsum(dims[:-1])) )
    theta = alpha, phi, Mo, alphaF, phiF

    return ln_likelihood_2SchfixedMF_densities(theta, Mr, xIDs, IDs, V, dims,
            offset, Mlim_right, DeltaM), Mr.size

def BIC_calc(filenames, NBinsM200, verbose):
    apString = '030kpc'
    DBICs = np.zeros((len(filenames), NBinsM200))

    for ifiles, fname in enumerate(filenames):
        with h5py.File(fname, 'r') as chain:
            for iMassBin in range(NBinsM200):
                group = 'mcmc_'+apString+'_bin-'+str(iMassBin)

                try:
                    lk1, n1 = lk_1sc_den(group, chain)
                    lk2, n2 = lk_2sc_den(group, chain)

                    # AIC1 = 2.*(3.-lk1)
                    # AIC2 = 2.*(5.-lk2)

                    try:
                        AIC1 = 2.*3.-2*lk1+(2.*3+1.)/(n1-3.-1.)
                        AIC2 = 2.*5.-2*lk2+(2.*5+1.)/(n2-5.-1.)
                    except ZeroDivisionError:
                        # I don't really use this, nor I put it into the output, so
                        # I give it a pass
                        pass

                    # if AIC1<AIC2:
                        # print('AIC prefers 1sc with delta=', AIC1-AIC2)
                        # print('file!! ', fname)
                    # else:
                        # print('AIC prefers 2sc with delta=', AIC1-AIC2)

                    BIC1 = np.log(n1)*3 - 2.*lk1
                    BIC2 = np.log(n2)*5 - 2.*lk2
                    DBIC = BIC1-BIC2
                    DBICs[ifiles, iMassBin] = DBIC

                    if verbose:
                        if DBIC>3.:
                            print('BIC prefers 2sc with delta=', DBIC, fname)
                        else:
                            print('BIC prefers 1sc with delta=', DBIC, fname)
                    # if BIC1>BIC2 and BIC1-BIC2<2.:
                        # if BIC1-BIC2<2.:
                            # print('BIC prefers 2sc since delta is non sigficative=', BIC1-BIC2)
                        # else:
                            # print('BIC prefers 2sc with delta=', BIC1-BIC2)
                    # else:
                        # print('BIC prefers 1sc with delta=', BIC1-BIC2)
                    # print(AIC1, AIC2)
                    # print(BIC1, BIC2)
                    # print(BIC1- BIC2)
                except (EmptyDataBin, dataSetNotFound):
                    DBICs[ifiles, iMassBin] = np.nan
                except KeyError:
                    print('filename = ', fname, ' group = ', group)
                    raise

    return DBICs


if __name__ ==  '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str, nargs='+',
            help='filenames')
    parser.add_argument('-v','--verbose',
            help='verbose output', 
            action='store_true')
    # parser.add_argument('-n', type=int,
            # help='numer of threads for computation [%(default)d]', default=12)

    # parser.add_argument('-C', type=int,
            # help='index of the cluster list [uses all the clusters]', default=-1,
            # choices=set(range(30)), nargs='+')

    args = parser.parse_args()

    BIC_calc(args.filenames, 3, args.verbose)
