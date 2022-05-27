

import numpy as np

from scipy.integrate import quad
from scipy.optimize import curve_fit

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pylab as plt
# from matplotlib import patches

# from .obs_lf import Popesso_2006r
# from .eagle_cosmology import EagleCosmology

from mcmc_fits.schechter_luminosityCuPy import f_single_schechternumpy as f_single
# from mcmc_fits.schechter_luminosity import f_double_schechter as f_double
# from mcmc_fits.schechter_luminosity import f_double_schechter_s as f_double2
from scipy.integrate import quad

import h5py

import time
import sys
import os
import subprocess
import astropy.table
import astropy.io
from pathlib import Path


# ================================================================================
def main(HaloDirectories, DATA_PATH, output_number,
        r200Intervals = [[0,1]], 
        lum_func_data_path = '.', ignore2Sc=False, add = False):

    global sepSingle
    global sepDouble

    STR_LEN = 123


    MAG_TYPE = 'EMILES_PDXX_DUST_CH_'
    # MAG_TYPE = 'EMILES_PDXX_NODUST_CH_'

    # print sepDouble
    # print sepDouble
    # print color.BOLD + 'z ='+OUTPUT_NUMBER + color.END


    # Apertures = [30, 50, 70]
    APERTURE_FIT = 30
    Apertures = [APERTURE_FIT]

    img_fmt = '.pdf'
    xx = np.arange(-26,-12,0.1)


    tabPath = os.path.join(DATA_PATH, lum_func_data_path , 'parameters')

    # read from the first file just to get NBinsM200
    NBinsM200 = read_NBinsM200(lum_func_data_path , MAG_TYPE,
                            output_number[0], r200Intervals[0], DATA_PATH)

    # in the case of no directory, create it
    if not os.path.isdir(os.path.join(outputPath, 'plots')):
            os.makedirs(os.path.join(outputPath, 'plots'))


    for ir200, nvir in enumerate(r200Intervals):

        finalFile = outputPath/ 'plots/shadedRegions.h5'

        if os.path.isfile(finalFile) and not add:
            raise IOError(str(finalFile) + ' is already present')

        # loop over redshift, a row in the final plot is at constant z
        # for iZ, (OUTPUT_NUMBER, Mbin) in enumerate(zip(output_number, res)):
        for iZ, OUTPUT_NUMBER  in enumerate(output_number):

            # M1 = Mbin[1]
            # M2 = Mbin[2]


            print('r200 BIN =' + str(nvir) +' z ='+OUTPUT_NUMBER)


            y0, x0, ey0, ex0, numGalaxies, normBinned,\
                    y0_r, x0_r, ey0_r, ex0_r, numGalaxies_r, \
                    y0_b, x0_b, ey0_b, ex0_b, \
                    numGalaxies_b, Apertures_data, zSnap, NBinsM200_tmp,  \
                    mag_min_fit, mag_min_fit_r, mag_min_fit_b \
                    = read_datafile(lum_func_data_path , MAG_TYPE,
                            OUTPUT_NUMBER, nvir, DATA_PATH)

            # for now just require this
            assert np.array(Apertures) == Apertures_data
            # -------------------------

            if NBinsM200_tmp != NBinsM200:
                raise ValueError('Number of bins in M200 is different from the first file read')

            # print(sepSingle)
            for iMassBin in range(NBinsM200):

                for iAp, aperture in enumerate(Apertures):

                    apString = '{:03d}'.format(aperture) + 'kpc'


                    # now plot the functions
                    if aperture == APERTURE_FIT:

                        # assuming that a mcmc has been performed
                        groupname = 'mcmc_'+apString+'_bin-'+str(iMassBin)

                        # read the data
                        string = format_r200(nvir)


                        alphamcmc, phimcmc, Momcmc, chain, \
                                alphaB,  phiB, MoB, \
                                alphaF,  phiF, MoF, chain2, ngal = \
                                read_mcmc(lum_func_data_path ,
                                    OUTPUT_NUMBER, nvir, DATA_PATH, groupname,
                                    ignore2Sc)

                        t = time.time()
                        y_min, y_max, y_min2, y_max2, median1, median2 = \
                                calc_shaded_region(chain, chain2, xx, 
                                        alphamcmc, phimcmc, Momcmc, 
                                        alphaB,  phiB, MoB, alphaF,  phiF, 
                                        MoF, ignore2Sc)
                        print('tot t=', time.time()-t)
                        if verbose:
                            print('writing ', finalFile)
                        with h5py.File(finalFile, 'a') as sh:
                            sh[OUTPUT_NUMBER+'/'+format_r200(nvir)
                                    +'/'+ str(iMassBin)+'/median1'] = median1
                            if not ignore2Sc:
                                sh[OUTPUT_NUMBER+'/'+  format_r200(nvir)
                                        +'/'+ str(iMassBin)+'/median2'] = median2
                            sh[OUTPUT_NUMBER+'/'+format_r200(nvir)
                                    +'/'+ str(iMassBin)+'/y_min'] = y_min
                            sh[OUTPUT_NUMBER+'/'+format_r200(nvir)
                                    +'/'+ str(iMassBin)+'/y_max'] = y_max
                            if not ignore2Sc:
                                sh[OUTPUT_NUMBER+'/'+format_r200(nvir)
                                        +'/'+ str(iMassBin)+'/y_max2'] = y_max2
                                sh[OUTPUT_NUMBER+'/'+format_r200(nvir)
                                        +'/'+ str(iMassBin)+'/y_min2'] = y_min2



    print('DONE!')


def calc_shaded_region(chain, chain2, x_plot, alpha, phi, Mo, alphaB,  phiB,
        MoB, alphaF,  phiF, MoF, ignore2Sc, calcDataSpace = True, useEntireChain = True):
    """
    Calculate the shaded region for 1 sigma values in the fit by using the
    chains in the mcmc.

    """
    import cupy as cp
    x = np.asarray(x_plot)
    xcp = cp.array(x)

    y_max  = np.full_like(x, -np.inf)
    y_min  = np.full_like(x,  np.inf)

    if not ignore2Sc:
        y_max2 = np.full_like(x, -np.inf)
        y_min2 = np.full_like(x,  np.inf)

    if calcDataSpace:
        ind = np.arange(chain.shape[0])
    else:
        # find all the triplets that stays in 1 sigma
        ind = np.flatnonzero(
                (chain[:,0] > (alpha[1] - alpha[0])) &
                (chain[:,0] < (alpha[1] + alpha[2])) &
                (chain[:,1] > (phi  [1] - phi  [0])) &
                (chain[:,1] < (phi  [1] + phi  [2])) &
                (chain[:,2] > (Mo   [1] - Mo   [0])) &
                (chain[:,2] < (Mo   [1] + Mo   [2])) )

    if ind.size > 0:
        if useEntireChain:
            choice = ind
        else:
            choice = np.random.choice(ind, min(10000, ind.size))
        if calcDataSpace:
            # samples = np.zeros((choice.size, x.size))
            # for ii, i in enumerate(choice):
                # samples[ii,:] = f_single(x, chain[i,0], chain[i, 1], chain[i,2])

            chaincp = cp.array(chain)
            c0 = cp.tile(chaincp[:,0],[x.size,1]).T
            c1 = cp.tile(chaincp[:,1],[x.size,1]).T
            c2 = cp.tile(chaincp[:,2],[x.size,1]).T
            x2 = cp.broadcast_to(xcp,(chain.shape[0],x.size))
            samples2 = f_single(x2, c0, c1, c2)

            # y_max = np.quantile(samples, 0.16, axis=0)
            # y_min = np.quantile(samples, 0.84, axis=0)
            # median1 = np.quantile(samples, 0.5, axis=0)
            y_maxcp = cp.percentile(samples2, 16., axis=0)
            y_mincp = cp.percentile(samples2, 84., axis=0)
            median1cp = cp.percentile(samples2, 50., axis=0)
            cp.cuda.Stream.null.synchronize()
            assert y_max.size == x.size
            assert y_min.size == x.size
            y_max  = cp.asnumpy(y_maxcp)
            y_min  = cp.asnumpy(y_mincp)
            median1 = cp.asnumpy(median1cp)

            
        else:
            for i in choice:
                y_max = np.maximum(y_max, f_single(x, chain[i,0], chain[i, 1], chain[i,2]))
                y_min = np.minimum(y_min, f_single(x, chain[i,0], chain[i, 1], chain[i,2]))

    # do the same for the double schecter

    if not ignore2Sc:
        if calcDataSpace:
            ind = np.arange(chain2.shape[0])
        else:
            # check if we are using the full double scheckter or the partial one, with
            # fixed Mo
            if chain2.shape[1] == 6:
                ind = np.flatnonzero(
                        (chain2[:,0] > (alphaB[1] - alphaB[0])) &
                        (chain2[:,0] < (alphaB[1] + alphaB[2])) &
                        (chain2[:,1] > (phiB  [1] - phiB  [0])) &
                        (chain2[:,1] < (phiB  [1] + phiB  [2])) &
                        (chain2[:,2] > (MoB   [1] - MoB   [0])) &
                        (chain2[:,2] < (MoB   [1] + MoB   [2])) &

                        (chain2[:,3] > (alphaF[1] - alphaF[0])) &
                        (chain2[:,3] < (alphaF[1] + alphaF[2])) &
                        (chain2[:,4] > (phiF  [1] - phiF  [0])) &
                        (chain2[:,4] < (phiF  [1] + phiF  [2])) &
                        (chain2[:,5] > (MoF   [1] - MoF   [0])) &
                        (chain2[:,5] < (MoF   [1] + MoF   [2])) )

            elif chain2.shape[1] == 5:
                ind = np.flatnonzero(
                        (chain2[:,0] > (alphaB[1] - alphaB[0])) &
                        (chain2[:,0] < (alphaB[1] + alphaB[2])) &
                        (chain2[:,1] > (phiB  [1] - phiB  [0])) &
                        (chain2[:,1] < (phiB  [1] + phiB  [2])) &
                        (chain2[:,2] > (MoB   [1] - MoB   [0])) &
                        (chain2[:,2] < (MoB   [1] + MoB   [2])) &

                        (chain2[:,3] > (alphaF[1] - alphaF[0])) &
                        (chain2[:,3] < (alphaF[1] + alphaF[2])) &
                        (chain2[:,4] > (phiF  [1] - phiF  [0])) &
                        (chain2[:,4] < (phiF  [1] + phiF  [2])) )

            else:
                raise ValueError

        if ind.size > 0:
            if useEntireChain:
                choice = ind
            else:
                choice = np.random.choice(ind, min(10000, ind.size))

            if calcDataSpace:
                # samples = np.zeros((choice.size, x.size))

                if chain2.shape[1] == 6:
                    for ii, i in enumerate(choice):
                        samples[ii,:] = f_single(x, chain2[i,0], chain2[i, 1],
                            chain2[i,2]) + f_single(x, chain2[i,3], chain2[i, 4], chain2[i,5])
                if chain2.shape[1] == 5:
                    # for ii, i in enumerate(choice):
                        # samples[ii,:] = f_single(x, chain2[i,0], chain2[i, 1],
                            # chain2[i,2]) + f_single(x, chain2[i,3], chain2[i, 4], chain2[i,2])
                    chain2cp = cp.array(chain2)
                    c0 = cp.tile(chain2cp[:,0],[x.size,1]).T
                    c1 = cp.tile(chain2cp[:,1],[x.size,1]).T
                    c2 = cp.tile(chain2cp[:,2],[x.size,1]).T
                    c3 = cp.tile(chain2cp[:,3],[x.size,1]).T
                    c4 = cp.tile(chain2cp[:,4],[x.size,1]).T
                    x2 = cp.broadcast_to(xcp,(chain2.shape[0],x.size))
                    samples2 = f_single(x2, c0, c1, c2) + f_single(x2, c3, c4, c2)
                    del c0, c1, c2, c3, c4
                    # assert np.all(np.isclose(samples, cp.asnumpy(samples2)))

                y_max2cp = cp.percentile(samples2, 16., axis=0)
                y_min2cp = cp.percentile(samples2, 84., axis=0)
                median2cp = cp.percentile(samples2,50., axis=0)
                cp.cuda.Stream.null.synchronize()
                assert y_max2.size == x.size
                assert y_min2.size == x.size
                y_max2  = cp.asnumpy(y_max2cp)
                y_min2  = cp.asnumpy(y_min2cp)
                median2 = cp.asnumpy(median2cp)

            else:
                if chain2.shape[1] == 6:

                    for i in choice:
                        y_max2= np.maximum(y_max2, f_single(x, chain2[i,0], chain2[i, 1],
                            chain2[i,2]) + f_single(x, chain2[i,3], chain2[i, 4], chain2[i,5]))
                        y_min2= np.minimum(y_min2, f_single(x, chain2[i,0], chain2[i, 1],
                            chain2[i,2]) + f_single(x, chain2[i,3], chain2[i, 4], chain2[i,5]))

                elif chain2.shape[1] == 5:

                    for i in choice:
                        y_max2= np.maximum(y_max2, f_single(x, chain2[i,0], chain2[i, 1],
                            chain2[i,2]) + f_single(x, chain2[i,3], chain2[i, 4], chain2[i,2]))
                        y_min2= np.minimum(y_min2, f_single(x, chain2[i,0], chain2[i, 1],
                            chain2[i,2]) + f_single(x, chain2[i,3], chain2[i, 4], chain2[i,2]))


    if ignore2Sc:
        return y_max, y_min, 0., 0., median1, 0.
    else:
        return y_max, y_min, y_max2, y_min2, median1, median2

def read_mcmc(haloDir, OUTPUT_NUMBER, nvir, DATA_PATH, groupname, ignore2Sc):
    """
    Read everything from a well documented file.

    """

    lfTotalFileName = os.path.join(DATA_PATH ,haloDir , 'chain' +
            OUTPUT_NUMBER + '_' + '_' +
            ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5')

    if verbose:
        print('reading from ', lfTotalFileName)

    with h5py.File(lfTotalFileName, 'r') as fileData:
        if density:
            group = fileData[groupname+'/1sc_den' ]
        else:
            group = fileData[groupname+'/1sc' ]
        alpha = group['results/alpha'][()]
        phi = group['results/phi'][()]
        Mo = group['results/Mo'][()]
        nburninDiscard = group.attrs['nburninDiscard']
        nsteps = group.attrs['nsteps']
        chain = group['samples'][()]
        nwalkers = group.attrs['nwalkers'][()]
        ndim = group.attrs['ndim'][()]
        try:
            num_gal1 = group['data_sample/Mr'].shape[0]
        except KeyError as e:
            num_gal1 = group['data_sample'].shape[0]

    # cut the chain and flatten it
    flat_cut_chain1sc = np.vstack(chain[nburninDiscard:, :, :])
    assert flat_cut_chain1sc.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)

    if not ignore2Sc:
        with h5py.File(lfTotalFileName, 'r') as fileData:
            if density:
                group = fileData[groupname+'/2scMoF_fixed_den' ]
            else:
                group = fileData[groupname+'/2scMoF_fixed' ]
            alphaB = group['results/alpha'][()]
            phiB = group['results/phi'][()]
            MoB = group['results/Mo'][()]
            alphaF = group['results/alphaF'][()]
            phiF = group['results/phiF'][()]
            MoF = group['results/MoF'][()]
            nburninDiscard = group.attrs['nburninDiscard']
            nsteps = group.attrs['nsteps']
            chain = group['samples'][()]
            nwalkers = group.attrs['nwalkers'][()]
            ndim = group.attrs['ndim'][()]
            try:
                num_gal2 = group['data_sample/Mr'].shape[0]
            except KeyError as e:
                num_gal2 = group['data_sample'].shape[0]

            # if not np.isclose(MoB[1], MoF[1]):
                # print (MoB[1], MoF[1])
                # raise ValueError

        # cut the chain and flatten it
        flat_cut_chain2sc = np.vstack(chain[nburninDiscard:, :, :])
        assert flat_cut_chain2sc.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)

        if num_gal2 != num_gal1:
            raise IOError('number of galaxies different in one and two fits')

        return alpha,  phi, Mo, flat_cut_chain1sc, \
                alphaB,  phiB, MoB, \
                alphaF,  phiF, MoF, flat_cut_chain2sc, num_gal1
    else:
        return alpha,  phi, Mo, flat_cut_chain1sc,\
                0., 0., 0., \
                0., 0., 0., 0., num_gal1


def read_datafile(haloDir, MAG_TYPE, OUTPUT_NUMBER, nvir, DATA_PATH):
    """
    Read everything from a well documented file.

    """

    lfTotalFileName = os.path.join(DATA_PATH ,haloDir , 'dataTotalLF_' + MAG_TYPE +
            OUTPUT_NUMBER + '_' + '_' +
            ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5')

    if verbose:
        print('reading from ', lfTotalFileName)

    with h5py.File(lfTotalFileName, 'r') as fileData:

        Apertures = fileData['Apertures'][()]
        NumberOfBins = fileData['NumberOfMagBins'][()]
        NumberOfClusterMassBins = fileData['NumberOfClusterMassBins'][()]
        zSnap = fileData['zSnap'][()]
        NBinsM200 = fileData['NumberOfClusterMassBins'][()]


        y     = fileData['total/y'][()]
        x     = fileData['total/x'][()]
        ey    = fileData['total/ey'][()]
        ex    = fileData['total/ex'][()]
        numGalaxies = fileData['total/numGalaxies'][()]
        mag_min_fit = fileData['total/mag_min_fit'][()]
        norm = fileData['total/norm'][()]

        y_b      = fileData['starForming/y'][()]
        x_b      = fileData['starForming/x'][()]
        ey_b     = fileData['starForming/ey'][()]
        ex_b     = fileData['starForming/ex'][()]
        numGalaxies_b  = fileData['starForming/numGalaxies'][()]
        mag_min_fit_b = fileData['starForming/mag_min_fit'][()]

        y_r      = fileData['quiescent/y'][()]
        x_r      = fileData['quiescent/x'][()]
        ey_r     = fileData['quiescent/ey'][()]
        ex_r     = fileData['quiescent/ex'][()]
        numGalaxies_r = fileData['quiescent/numGalaxies'][()]
        mag_min_fit_r = fileData['quiescent/mag_min_fit'][()]

    return (y, x, ey, ex, numGalaxies , norm,
            y_r, x_r, ey_r, ex_r, numGalaxies_r,
            y_b, x_b, ey_b, ex_b, numGalaxies_b, Apertures, zSnap, NBinsM200,
            mag_min_fit, mag_min_fit_r, mag_min_fit_b)

def format_r200(nvir):
    # return ('r200_' + '{:02d}'.format(nvir[0]) + '-' +
            # ('{0:1.1f}'.format(nvir[1])).replace(".","p"))
    return ('r200_' + ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p"))


def plot_lum_function(ax, x0, y0, ex0, ey0, apString, colors, mfc_colors, labels, fmt,
        iMassBin, zorder, norm, min_mag = 0.):
    nonzero = (y0[apString,:,iMassBin] > 0) & (x0[apString,:,iMassBin] <
            min_mag)
    x = x0[apString,nonzero,iMassBin]
    y = y0[apString,nonzero,iMassBin]
    xerr = ex0[apString,nonzero,iMassBin]
    yerr = ey0[apString,nonzero,iMassBin]

    y*=norm
    yerr*=norm

    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
            fmt=fmt, mec=colors, mfc=mfc_colors, ecolor=colors,
            zorder=zorder, label=labels, markersize=8, elinewidth=2)

    return x, y, yerr


def read_NBinsM200(haloDir, MAG_TYPE, OUTPUT_NUMBER, nvir, DATA_PATH):
    """
    Just read NBinsM200
    """

    lfTotalFileName = os.path.join(DATA_PATH ,haloDir , 'dataTotalLF_' + MAG_TYPE +
            OUTPUT_NUMBER + '_' + '_' +
            ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5')

    if verbose:
        print('reading from ', lfTotalFileName)

    with h5py.File(lfTotalFileName, 'r') as fileData:

        NBinsM200 = fileData['NumberOfClusterMassBins'][()]


    return NBinsM200

# ================================================================================

if __name__ == '__main__':
    import argparse
    from lum_functions.r200_intervals import r200IntervalsOriginal

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose',
            help='verbose output',
            action='store_true')

    parser.add_argument('pathData',
            help='path of the luminosity functions', type=str)

    parser.add_argument('-a','--add',
            help='add new data to existing file',
            action='store_true')
    parser.add_argument('--ignore2Sc',
            help='ignore two component Schecther',
            action='store_true')
    parser.add_argument('-r','--radius', nargs='+',
            help='index of the radius bin', type=int,
            choices=set(range(len(r200IntervalsOriginal))))
    parser.add_argument('-S', type=int,
            help='snapshot to process [1-30]', 
            default=list(range(1,30)), choices = range(30), nargs='+')
    args = parser.parse_args()

    verbose = args.verbose


    data = ['CE-0/HYDRO/data',
            'CE-1/HYDRO/data',
            'CE-2/HYDRO/data',
            'CE-3/HYDRO/data',
            'CE-4/HYDRO/data',
            'CE-5/HYDRO/data',
            'CE-6/HYDRO/data',
            'CE-7/HYDRO/data',
            'CE-8/HYDRO/data',
            'CE-9/HYDRO/data',
            'CE-10/HYDRO/data',
            'CE-11/HYDRO/data',
            'CE-12/HYDRO/data',
            'CE-13/HYDRO/data',
            'CE-14/HYDRO/data',
            'CE-15/HYDRO/data',
            'CE-16/HYDRO/data',
            'CE-17/HYDRO/data',
            'CE-18/HYDRO/data',
            'CE-19/HYDRO/data',
            'CE-20/HYDRO/data',
            'CE-21/HYDRO/data',
            'CE-22/HYDRO/data',
            'CE-23/HYDRO/data',
            'CE-24/HYDRO/data',
            'CE-25/HYDRO/data',
            'CE-26/HYDRO/data',
            'CE-27/HYDRO/data',
            'CE-28/HYDRO/data',
            'CE-29/HYDRO/data']

    output_number_all = [
            '000_z014p003',
            '001_z006p772',
            '002_z004p614',
            '003_z003p512',
            '004_z002p825',
            '005_z002p348',
            '006_z001p993',
            '007_z001p716',
            '008_z001p493',
            '009_z001p308',
            '010_z001p151',
            '011_z001p017',
            '012_z000p899',
            '013_z000p795',
            '014_z000p703',
            '015_z000p619',
            '016_z000p543',
            '017_z000p474',
            '018_z000p411',
            '019_z000p366',
            '020_z000p352',
            '021_z000p297',
            '022_z000p247',
            '023_z000p199',
            '024_z000p155',
            '025_z000p113',
            '026_z000p101',
            '027_z000p073',
            '028_z000p036',
            '029_z000p000'
            ]


    args.S.sort()
    output_number = [output_number_all[i]  for i in args.S]
    # when all of them are selected, I like it inverted
    output_number.reverse()

    r200Intervals = [[0,0.5], [0,1], [0,2], [0,3],
            [1,2], [1,3], [1,5], [3,5], [5,10], [0,10]]
    # r200Intervals = [[0,0.5], [0,1], [5,10],[0,10]]
    # r200Intervals = [[5,10], [0,10]]
    # r200Intervals = [ [0,0.5], [0,1], [5,10], [0,10]]
    r200Intervals = [[0,0.5], [0,1], [5,10]]
    r200Intervals = [[0,1]]
    # r200Intervals = [[0,0.5], [0,1], [0,2], [5,10], [0,10]]
    # r200Intervals = [[5,10]]

    if args.radius is not None:
        r200Intervals = [r200IntervalsOriginal[i] for i in args.radius]
    else:
        r200Intervals = r200IntervalsOriginal


    lum_func_data_path = 'lum_function_data'
    LUM_FUNC_PATH = args.pathData
    outputPath  = Path(LUM_FUNC_PATH)

    # plot  the number density of galaxies, not the number
    density = True


    main(data, LUM_FUNC_PATH, output_number,
            r200Intervals,
            lum_func_data_path = lum_func_data_path, add=args.add,
            ignore2Sc = args.ignore2Sc,
            )
