

import numpy as np

from scipy.integrate import quad
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from matplotlib import patches

# from mcmc_fits.schechter_luminosity import f_double_schechter_popesso_2006 as f_double_popesso_2006
from mcmc_fits.schechter_luminosity import f_single_schechter as f_single
from mcmc_fits.schechter_luminosity import f_double_schechter as f_double
from mcmc_fits.schechter_luminosity import f_double_schechter_s as f_double2
from scipy.integrate import quad

from multiprocessing import Array

import h5py

import time
import sys
import os
import subprocess
import astropy.table
import astropy.io
from astropy.io import ascii

plt.ioff()


class EmptyDataBin(Exception):
    pass
# ================================================================================
def main(HaloDirectories, DATA_PATH, output_number,
        r200Intervals = [[0,1]],
        lum_func_data_path = '.', overwrite = False,
        plot_fit_binned = False, plot_bayes = 0,
        plotB = False, plotR = False, useDensities = False, phiConst = 1.):

    global sepSingle
    global sepDouble

    STR_LEN = 123

    sepSingle = color.RED + '-' * STR_LEN + color.END
    sepDouble = color.RED + '=' * STR_LEN + color.END

    # OUTPUT_NUMBER = '029_z000p000'
    # OUTPUT_NUMBER = '026_z000p101'

    MAG_TYPE = 'EMILES_PDXX_DUST_CH_'
    # MAG_TYPE = 'EMILES_PDXX_NODUST_CH_'

    normalize = False

    # print sepDouble
    # print sepDouble
    # print color.BOLD + 'z ='+OUTPUT_NUMBER + color.END


    APERTURE_FIT = 30
    Apertures = [APERTURE_FIT]
    assert len(Apertures) == 1

    img_fmt = '.dat'


    # lum_func_data_path = 'tmp'
    tabPath = os.path.join(DATA_PATH, lum_func_data_path , 'parameters')

    # read from the first file just to get NBinsM200
    NBinsM200 = read_NBinsM200(lum_func_data_path , MAG_TYPE,
                            output_number[0], r200Intervals[0], DATA_PATH)


    for ir200, nvir in enumerate(r200Intervals):


        names = ['M200bin', 'snap', 'z', 'N',
                'Mr', 'Mr_p', 'Mr_m',
                'alpha', 'alpha_m', 'alpha_p',
                'phi', 'phi_m','phi_p',
                'int_19', 'chi2_red_1sc', 
                'Mr_b', 'Mr_b_m', 'Mr_b_p',
                'alpha_b', 'alpha_b_m','alpha_b_p',
                'phi_b', 'phi_b_m','phi_b_p',
                'Mr_f', 'Mr_f_m', 'Mr_f_p',
                'alpha_f', 'alpha_f_m','alpha_f_p',
                'phi_f',  'phi_f_m','phi_f_p',
                'int2_19', 'chi2_red_2sc', 'richness', 
                'Mr_MF_b', 'Mr_MF_b_m', 'Mr_MF_b_p',
                'alpha_MF_b', 'alpha_MF_b_m','alpha_MF_b_p',
                'phi_MF_b', 'phi_MF_b_m','phi_MF_b_p',
                'alpha_MF_f', 'alpha_MF_f_m','alpha_MF_f_p',
                'phi_MF_f',  'phi_MF_f_m','phi_MF_f_p', 'intMF_19', 
                'chi2_red_2scMF', 'Mlim']

        dtype = ('i',)*2 + ('f8',) + ('i',) + ('f8',)*31+('i',)+('f8',)*18
        tabT = astropy.table.Table(names = names, dtype=dtype, masked=True)
        tabR = astropy.table.Table(names = names, dtype=dtype, masked=True)
        tabB = astropy.table.Table(names = names, dtype=dtype, masked=True)

        # latex tables, with pretty strings ^ and _
        names = ['M200bin', 'z', 'N','richness',
                r'$M_\mathrm{r}^*$',
                r'$\alpha$',
                r'$\phi$',
                'int_19',
                r'$M_\mathrm{r,b}^*$',
                r'$\alpha_b$',
                r'$\phi_b$',
                r'$M_\mathrm{r,f}^*$',
                r'$\alpha_\mathrm{f}$',
                r'$\phi_\mathrm{f}$',
                'int2_19',
                r'$M_\mathrm{r,mf}^*$',
                r'$\alpha_bmf$',
                r'$\phi_bmf$',
                r'$\alpha_\mathrm{f, mf}$',
                r'$\phi_\mathrm{f, mf}$',
                'int3_19']

        dtype = ('i',) + ('f8',) + ('i',)*2 + ('S100',)*3 \
                +('f8',)+('S100',)*6+('f8',)+('S100',)*5+('f8',)
        tabTL = astropy.table.Table(names = names, dtype=dtype)
        tabRL = astropy.table.Table(names = names, dtype=dtype)
        tabBL = astropy.table.Table(names = names, dtype=dtype)


        # loop over redshift, a row in the final plot is at constant z
        for iZ, OUTPUT_NUMBER in enumerate(output_number):


            print(sepDouble)
            print(color.BOLD + 'r200 BIN =' + str(nvir) +' z ='+OUTPUT_NUMBER
                    + color.END)


            y0, x0, ey0, ex0, numGalaxies, \
                    y0_r, x0_r, ey0_r, ex0_r, numGalaxies_r, \
                    y0_b, x0_b, ey0_b, ex0_b, \
                    numGalaxies_b, Apertures_data, zSnap, NBinsM200_tmp,  \
                    mag_min_fit, mag_min_fit_r, mag_min_fit_b, \
                    norm_binn, norm_r_binn, norm_b_binn \
                    = read_datafile(lum_func_data_path , MAG_TYPE,
                            OUTPUT_NUMBER, nvir, DATA_PATH)

            # for now just require this
            assert np.array(Apertures) == Apertures_data
            # -------------------------

            if NBinsM200_tmp != NBinsM200:
                raise ValueError('Number of bins in M200 is different from the first file read')


            print(sepSingle)
            for iMassBin in range(NBinsM200):
                # loop over the bins in M200 of the cluster, this will be our
                # row index

                for iAp, aperture in enumerate(Apertures):

                    apString = '{:03d}'.format(aperture) + 'kpc'


                    # assuming that a mcmc has been performed
                    groupname = 'mcmc_'+apString+'_bin-'+str(iMassBin)


                    tabT, tabTL = tables(lum_func_data_path , OUTPUT_NUMBER, nvir, DATA_PATH,
                            groupname, y0, x0, ey0, norm_binn, iAp, iMassBin,
                            zSnap, tabT, tabTL, phiConst)

                    # tabR, tabRL = tables(lum_func_data_path , OUTPUT_NUMBER, nvir, DATA_PATH,
                            # groupname+'_r', y0_r, x0_r, ey0_r, norm_r_binn, 
                            # iAp, iMassBin, zSnap, tabR, tabRL)

                    # tabB, tabRL = tables(lum_func_data_path , OUTPUT_NUMBER, nvir, DATA_PATH,
                            # groupname+'_b', y0_b, x0_b, ey0_b, norm_r_binn, 
                            # iAp, iMassBin, zSnap, tabB, tabRL)

        lfTotalFileNames = [os.path.join(DATA_PATH ,lum_func_data_path , 'chain' +
                OUTPUT_NUMBER + '_' + '_' +
                # '{:02d}'.format(nvir[0]) + '-' +
                ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
                ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5') for 
                OUTPUT_NUMBER in output_number]

        from mcmc_fits.calc_likelihood import BIC_calc
        DBICs = BIC_calc(lfTotalFileNames, NBinsM200, False)
        DBICs = DBICs.flatten()

        # in the case of no directory, create it
        if not os.path.isdir(tabPath):
                os.makedirs(tabPath)

        plotFilename = os.path.join(tabPath, 'tabMCMC_' +
                format_r200(nvir) )

        if verbose:
            print('saving ', plotFilename)

        tabT = model_selection(tabT)
        tabB = model_selection(tabB)
        tabR = model_selection(tabR)

        tabT['DeltaBIC'] = DBICs
        tabTL['$\Delta BIC$'] = DBICs
        tabTL['$\M_\mathrm{f}$'] = tabT['Mlim']
        # tabTL['$\Delta BIC$'].dtype = 
        # print(dir(tabTL['$\Delta BIC$']))

        put_units(tabT)
        put_units(tabB)
        put_units(tabR)
        
        tabT.write(plotFilename+ img_fmt, overwrite = overwrite, format='ascii.ipac')
        tabB.write(plotFilename+ '_b'+img_fmt, overwrite = overwrite, format='ascii.ipac')
        tabR.write(plotFilename+ '_r'+img_fmt, overwrite = overwrite, format='ascii.ipac')

        exclude_latex = ['int_19', 'int2_19', 'M200bin']

        # exclude 2 sc
        exclude_latex = ['int_19', 'int2_19', 'M200bin', 
                r'$M_\mathrm{r,b}^*$', r'$\alpha_b$', r'$\phi_b$',
                r'$M_\mathrm{r,f}^*$', r'$\alpha_\mathrm{f}$',
                r'$\phi_\mathrm{f}$', 'int3_19']
        # exclude 2 sc with shared knee
        # exclude_latex.extend([
                    # r'$M_\mathrm{r,mf}^*$',
                    # r'$\alpha_bmf$',
                    # r'$\phi_bmf$',
                    # r'$\alpha_\mathrm{f, mf}$',
                    # r'$\phi_\mathrm{f, mf}$'])


        tabTL.write(plotFilename+ '.tex', overwrite = overwrite,
                format='latex', formats={'z': '%0.2f', names[4]: '{:>110s}', 
                    '$\Delta BIC$' : '%0.2f', '$\M_\mathrm{f}$' : '%0.2f'},
                exclude_names=exclude_latex ,
                fill_values=[(ascii.masked, '--')])

        tabRL.write(plotFilename+ '_r.tex', overwrite = overwrite,
                format='latex', formats={'z': '%0.2f', 'z': '%0.2f'},
                exclude_names=exclude_latex ,
                fill_values=[(ascii.masked, '--')])

        tabBL.write(plotFilename+ '_b.tex', overwrite = overwrite,
                format='latex', formats={'z': '%0.2f', 'z': '%0.2f'},
                exclude_names = exclude_latex,
                fill_values=[(ascii.masked, '--')])

    print(sepDouble)
    print(color.BOLD + 'DONE!' + color.END)
    print(sepDouble)




def model_selection(tabT):
    """
    Do the model selection.

    """

    typefit = np.full_like(tabT['chi2_red_1sc'][:], 1)
    mask = np.copy(tabT['chi2_red_2scMF'].mask)

    # galaxies here should be flagged as 2
    ind2= np.abs(tabT['chi2_red_1sc'][:] -1.) > \
            np.abs(tabT['chi2_red_2scMF'][:] -1.)

    # max_err = np.maximum(tabT['alpha_f_m'], tabT['alpha_f_p'])
    # max_err2 = np.maximum(tabT['alpha_b_m'], tabT['alpha_b_p'])

    max_err = np.maximum(tabT['alpha_MF_f_m'], tabT['alpha_MF_f_p'])
    max_err2 = np.maximum(tabT['alpha_MF_b_m'], tabT['alpha_MF_b_p'])

    # galaxies here should be flagged as 1
    ind1 = np.logical_or(np.abs(max_err / tabT['alpha_MF_f']) >
            0.4, np.abs(max_err2 / tabT['alpha_MF_b']) > 0.4)
    typefit[np.logical_and(ind2, np.logical_not(ind1))] = 2


    newcol = astropy.table.Column(typefit, name='typeFit', dtype='i')
    newcol [mask] = 1
    tabT.add_column(newcol)

    return tabT



def tables(lum_func_data_path , OUTPUT_NUMBER, nvir, DATA_PATH,
        groupname, y0, x0, ey0, norm_binn, iAp, iMassBin, zSnap, tabT,
        tabTL,const ):

    alphamcmc, phimcmc, Momcmc, chain, \
            alphaB,  phiB, MoB, \
            alphaF,  phiF, MoF, chain2, \
            ngal, Mlim_right, mask, data_sample, \
            alphaMFB,  phiMFB, MoMFB, \
            alphaMFF,  phiMFF, MoMFF, chain3 = \
            read_mcmc(lum_func_data_path ,
                    OUTPUT_NUMBER, nvir, DATA_PATH, groupname)
    if verbose:
        print ('read ', lum_func_data_path)

    # # manually force the 2 Schechter fit to be ignored
    # # useful for tables
    # mask[1] = False

    # richness
    richness = np.count_nonzero(data_sample < Momcmc[1]+2.)

    # normalization of the total one
    if not mask[0]:
        norm = quad(f_single, -30., -19.,
                args=(alphamcmc[1], phimcmc[1], Momcmc[1]))[0]
    else:
        norm = 0.

    if not mask[1]:
        norm2 = quad(f_double2, -30., -19.,
                args=(alphaB[1], alphaF[1], phiB[1],phiF[1],
                    MoB[1], MoF[1]))[0]
    else:
        norm2 = 0.

    if not mask[2]:
        norm3 = quad(f_double2, -30., -19.,
                args=(alphaMFB[1], alphaMFF[1], phiMFB[1],phiMFF[1],
                    MoMFB[1], MoMFF[1]))[0]
    else:
        norm3 = 0.

    x = x0[iAp,:,iMassBin]
    dx = np.diff(x)[0]
    assert np.all(np.isclose(np.diff(x), dx))
    del x

    nonzero = (y0[iAp,:,iMassBin] > 0) & (x0[iAp,:,iMassBin] <
            Mlim_right)
    x = x0[iAp,nonzero,iMassBin]
    y = y0[iAp,nonzero,iMassBin]
    yerr = ey0[iAp,nonzero,iMassBin]

    # recover the non normalized histogram. NOTE: norm_binn
    # contains also the width of the bin
    y *= norm_binn[iMassBin]/dx 
    yerr *= norm_binn[iMassBin]/dx 


    if not mask[0]:
        res_single = ((y - f_single(x, alphamcmc[1], 
            phimcmc[1], Momcmc[1])) / np.abs(yerr))
    else:
        res_single = np.zeros_like(y)

    if not mask[1]:
        res_double = ((y - f_double2(x,
            alphaB[1], alphaF[1], phiB[1],phiF[1], MoB[1], MoF[1]))
            / np.abs(yerr))
    else:
        res_double = np.zeros_like(y)

    if not mask[2]:
        res_doubleMF = ((y - f_double2(x,
            alphaMFB[1], alphaMFF[1], phiMFB[1],phiMFF[1], MoMFB[1], MoMFF[1]))
            / np.abs(yerr))
    else:
        res_doubleMF = np.zeros_like(y)


    chisquare_single   = np.sum(res_single**2  )/(res_single.size-3)
    chisquare_double   = np.sum(res_double**2  )/(res_double.size-5)
    chisquare_doubleMF = np.sum(res_doubleMF**2)/(res_doubleMF.size-4)

    if mask[0] or mask[1] or mask[2]:
        tabT = astropy.table.Table(tabT, masked=True)
        tabTL = astropy.table.Table(tabTL, masked=True)

    tabT.add_row([iMassBin, int(OUTPUT_NUMBER[0:3]), zSnap, ngal,
        Momcmc[1], Momcmc[0], Momcmc[2],
        alphamcmc[1],alphamcmc[0],alphamcmc[2],
        phimcmc[1], phimcmc[0], phimcmc[2],
        norm, chisquare_single,
        MoB[1],MoB[0], MoB[2],
        alphaB[1],alphaB[0],alphaB[2],
        phiB[1],phiB[0],phiB[2],
        MoF[1],MoF[0],MoF[2],
        alphaF[1],alphaF[0],alphaF[2],
        phiF[1],phiF[0], phiF[2],
        norm2, chisquare_double, richness, 
        MoMFB[1],MoMFB[0], MoMFB[2],
        alphaMFB[1],alphaMFB[0],alphaMFB[2],
        phiMFB[1],phiMFB[0],phiMFB[2],
        alphaMFF[1],alphaMFF[0],alphaMFF[2],
        phiMFF[1],phiMFF[0], phiMFF[2], norm3, 
        chisquare_doubleMF, Mlim_right])

    if mask[1]:

        # mask the second schecter
        tabT.mask['Mr_b'][-1] = True
        tabT.mask['Mr_b_m'][-1] = True
        tabT.mask['Mr_b_p'][-1] = True

        tabT.mask['alpha_b'][-1] = True
        tabT.mask['alpha_b_m'][-1] = True
        tabT.mask['alpha_b_p'][-1] = True

        tabT.mask['phi_b'][-1] = True
        tabT.mask['phi_b_m'][-1] = True
        tabT.mask['phi_b_p'][-1] = True

        tabT.mask['Mr_f'][-1] = True
        tabT.mask['Mr_f_m'][-1] = True
        tabT.mask['Mr_f_p'][-1] = True

        tabT.mask['alpha_f'][-1] = True
        tabT.mask['alpha_f_m'][-1] = True
        tabT.mask['alpha_f_p'][-1] = True

        tabT.mask['phi_f'][-1] = True
        tabT.mask['phi_f_m'][-1] = True
        tabT.mask['phi_f_p'][-1] = True

        tabT.mask['int2_19'][-1] = True
        tabT.mask['chi2_red_2sc'][-1] = True

    if mask[2]:

        # mask the second schecter
        tabT.mask['Mr_MF_b'][-1] = True
        tabT.mask['Mr_MF_b_m'][-1] = True
        tabT.mask['Mr_MF_b_p'][-1] = True

        tabT.mask['alpha_MF_b'][-1] = True
        tabT.mask['alpha_MF_b_m'][-1] = True
        tabT.mask['alpha_MF_b_p'][-1] = True

        tabT.mask['phi_MF_b'][-1] = True
        tabT.mask['phi_MF_b_m'][-1] = True
        tabT.mask['phi_MF_b_p'][-1] = True

        try:
            tabT.mask['Mr_MF_f'][-1] = True
            tabT.mask['Mr_MF_f_m'][-1] = True
            tabT.mask['Mr_MF_f_p'][-1] = True
        except KeyError:
            pass

        tabT.mask['alpha_MF_f'][-1] = True
        tabT.mask['alpha_MF_f_m'][-1] = True
        tabT.mask['alpha_MF_f_p'][-1] = True

        tabT.mask['phi_MF_f'][-1] = True
        tabT.mask['phi_MF_f_m'][-1] = True
        tabT.mask['phi_MF_f_p'][-1] = True

        tabT.mask['intMF_19'][-1] = True
        tabT.mask['chi2_red_2scMF'][-1] = True


    if mask[0]:

        tabT.mask['Mr'][-1] = True
        tabT.mask['Mr_m'][-1] = True
        tabT.mask['Mr_p'][-1] = True

        tabT.mask['alpha'][-1] = True
        tabT.mask['alpha_m'][-1] = True
        tabT.mask['alpha_p'][-1] = True

        tabT.mask['phi'][-1] = True
        tabT.mask['phi_m'][-1] = True
        tabT.mask['phi_p'][-1] = True

    tabTL.add_row([iMassBin, zSnap, ngal, richness,
        '${0:6.2f}'.format(Momcmc[1])+'^{+' +
        '{0:5.2f}'.format(Momcmc[2])+'}'+'_{-' +
        '{0:5.2f}'.format(Momcmc[0])+'}$',

        '${0:6.2f}'.format(alphamcmc[1])+'^{+' +
        '{0:5.2f}'.format(alphamcmc[2])+'}'+'_{-' +
        '{0:5.2f}'.format(alphamcmc[0])+'}$',

        '${0:6.2f}'.format(phimcmc[1]*const)+'^{+' +
        '{0:5.2f}'.format(phimcmc[2]*const)+'}'+'_{-' +
        '{0:5.2f}'.format(phimcmc[0]*const)+'}$',

        norm,

        '${0:6.2f}'.format(MoB[1])+'^{+' +
        '{0:5.2f}'.format(MoB[2])+'}'+'_{-' +
        '{0:5.2f}'.format(MoB[0])+'}$',

        '${0:6.2f}'.format(alphaB[1])+'^{+' +
        '{0:5.2f}'.format(alphaB[2])+'}'+'_{-' +
        '{0:5.2f}'.format(alphaB[0])+'}$',

        '${0:6.2f}'.format(phiB[1]*const)+'^{+' +
        '{0:5.2f}'.format(phiB[2]*const)+'}'+'_{-' +
        '{0:5.2f}'.format(phiB[0]*const)+'}$',

        '${0:6.2f}'.format(MoF[1])+'^{+' +
        '{0:5.2f}'.format(MoF[2])+'}'+'_{-' +
        '{0:5.2f}'.format(MoF[0])+'}$',

        '${0:6.2f}'.format(alphaF[1])+'^{+' +
        '{0:5.2f}'.format(alphaF[2])+'}'+'_{-' +
        '{0:5.2f}'.format(alphaF[0])+'}$',

        '${0:6.2f}'.format(phiF[1]*const)+'^{+' +
        '{0:5.2f}'.format(phiF[2]*const)+'}'+'_{-' +
        '{0:5.2f}'.format(phiF[0]*const)+'}$',

        norm2,
        
        '${0:6.2f}'.format(MoMFB[1])+'^{+' +
        '{0:5.2f}'.format(MoMFB[2])+'}'+'_{-' +
        '{0:5.2f}'.format(MoMFB[0])+'}$',

        '${0:6.2f}'.format(alphaMFB[1])+'^{+' +
        '{0:5.2f}'.format(alphaMFB[2])+'}'+'_{-' +
        '{0:5.2f}'.format(alphaMFB[0])+'}$',

        '${0:6.2f}'.format(phiMFB[1]*const)+'^{+' +
        '{0:5.2f}'.format(phiMFB[2]*const)+'}'+'_{-' +
        '{0:5.2f}'.format(phiMFB[0]*const)+'}$',

        '${0:6.2f}'.format(alphaMFF[1])+'^{+' +
        '{0:5.2f}'.format(alphaMFF[2])+'}'+'_{-' +
        '{0:5.2f}'.format(alphaMFF[0])+'}$',

        '${0:6.2f}'.format(phiMFF[1]*const)+'^{+' +
        '{0:5.2f}'.format(phiMFF[2]*const)+'}'+'_{-' +
        '{0:5.2f}'.format(phiMFF[0]*const)+'}$',

        norm3])

    if mask[2]:
        tabTL.mask['$M_\mathrm{r,mf}^*$'][-1] = True
        tabTL.mask[r'$\alpha_bmf$'][-1] = True
        tabTL.mask[r'$\phi_bmf$'][-1] = True
        tabTL.mask[r'$\alpha_\mathrm{f, mf}$'][-1] = True
        tabTL.mask[r'$\phi_\mathrm{f, mf}$'][-1] = True

    if mask[1]:
        tabTL.mask['$M_\mathrm{r,b}^*$'][-1] = True
        tabTL.mask[r'$\alpha_b$'][-1] = True
        tabTL.mask[r'$\phi_b$'][-1] = True
        tabTL.mask[r'$M_\mathrm{r,f}^*$'][-1] = True
        tabTL.mask[r'$\alpha_\mathrm{f}$'][-1] = True
        tabTL.mask[r'$\phi_\mathrm{f}$'][-1] = True

    if mask[0]:
        tabTL.mask[r'$M_\mathrm{r}^*$'][-1] = True
        tabTL.mask[r'$\alpha$'][-1] = True
        tabTL.mask[r'$\phi$'][-1] = True

    return tabT, tabTL

def read_mcmc(haloDir, OUTPUT_NUMBER, nvir, DATA_PATH, groupname):
    """
    Read everything from a well documented file.

    """

    lfTotalFileName = os.path.join(DATA_PATH ,haloDir , 'chain' +
            OUTPUT_NUMBER + '_' + '_' +
            # '{:02d}'.format(nvir[0]) + '-' +
            ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5')

    if verbose:
        print ('reading from ', lfTotalFileName)

    mask = [False, False, False]

    try:
        with h5py.File(lfTotalFileName, 'r') as fileData:
            if useDensities:
                group = fileData[groupname+'/1sc_den' ]
            else:
                group = fileData[groupname+'/1sc' ]

            num_gal1 = group['data_sample/IDs'].size
            if num_gal1 == 0:
                raise EmptyDataBin

            alpha = group['results/alpha'][()]
            phi = group['results/phi'][()]
            Mo = group['results/Mo'][()]
            nburninDiscard = group.attrs['nburninDiscard']
            nsteps = group.attrs['nsteps']
            chain = group['samples'][()]
            nwalkers = group.attrs['nwalkers'][()]
            ndim = group.attrs['ndim'][()]
            data_sample = group['data_sample/Mr'][()]
            Mlim_right = group.attrs['Mlim_right'][()]

        # cut the chain and flatten it
        flat_cut_chain1sc = np.vstack(chain[nburninDiscard:, :, :])
        assert flat_cut_chain1sc.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)
    except EmptyDataBin as e:
        mask[0] = True
        num_gals1 = 0

        # just zeros
        flat_cut_chain1sc = 0.
        alpha,  phi, Mo = [[0.]*3]*3
        Mlim_right = 0.
        data_sample = np.array([], dtype=np.float)

    try:
        with h5py.File(lfTotalFileName, 'r') as fileData:
            group = fileData[groupname+'/2sc' ]
            num_gal2 = group['data_sample/IDs'].size
            if num_gal2 == 0:
                raise EmptyDataBin
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
            Mlim_right2 = group.attrs['Mlim_right'][()]
            data_sample2 = group['data_sample/Mr'][()]

        # cut the chain and flatten it
        flat_cut_chain2sc = np.vstack(chain[nburninDiscard:, :, :])
        assert flat_cut_chain2sc.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)
        assert Mlim_right == Mlim_right2

        if num_gal2 != num_gal1:
            raise IOError('number of galaxies different in one and two fits')

        if not np.array_equal(data_sample, data_sample2):
            raise IOError('galaxies samples different in one and two fits')

    except (KeyError, EmptyDataBin) as e:
        mask[1] = True
        flat_cut_chain2sc = 0.
        alphaF,  phiF, MoF, alphaB,  phiB, MoB = [[0.]*3]*6

    try:
        with h5py.File(lfTotalFileName, 'r') as fileData:
            if useDensities:
                group = fileData[groupname+'/2scMoF_fixed_den' ]
            else:
                group = fileData[groupname+'/2scMoF_fixed' ]
            num_gal3 = group['data_sample/IDs'].size
            if num_gal3 == 0:
                raise EmptyDataBin
            alphaMFB = group['results/alpha'][()]
            phiMFB = group['results/phi'][()]
            MoMFB = group['results/Mo'][()]
            alphaMFF = group['results/alphaF'][()]
            phiMFF = group['results/phiF'][()]
            MoMFF = group['results/MoF'][()]
            nburninDiscard = group.attrs['nburninDiscard']
            nsteps = group.attrs['nsteps']
            chain = group['samples'][()]
            nwalkers = group.attrs['nwalkers'][()]
            ndim = group.attrs['ndim'][()]
            Mlim_right3 = group.attrs['Mlim_right'][()]
            data_sample3 = group['data_sample/Mr'][()]

        # cut the chain and flatten it
        flat_cut_chain2scMF = np.vstack(chain[nburninDiscard:, :, :])
        assert flat_cut_chain2scMF.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)
        assert Mlim_right == Mlim_right3

        if num_gal3 != num_gal1:
            raise IOError('number of galaxies different in one and two fits')

        if not np.array_equal(data_sample, data_sample3):
            raise IOError('galaxies samples different in one and two fits')

    except (KeyError, EmptyDataBin) as e:
        mask[2] = True
        flat_cut_chain2scMF = 0.
        alphaMFF,  phiMFF, MoMFF, alphaMFB,  phiMFB, MoMFB = [[0.]*3]*6

    return alpha,  phi, Mo, flat_cut_chain1sc, \
            alphaB,  phiB, MoB, \
            alphaF,  phiF, MoF, flat_cut_chain2sc, num_gal1, \
            Mlim_right, mask, data_sample, \
            alphaMFB,  phiMFB, MoMFB, \
            alphaMFF,  phiMFF, MoMFF, flat_cut_chain2scMF


def format_r200(nvir):
    return ('r200_' + 
            ('{0:1.1f}'.format(nvir[0])).replace(".","p")
            + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p"))



def read_NBinsM200(haloDir, MAG_TYPE, OUTPUT_NUMBER, nvir, DATA_PATH):
    """
    Just read NBinsM200
    """

    lfTotalFileName = os.path.join(DATA_PATH ,haloDir , 'dataTotalLF_' + MAG_TYPE +
            OUTPUT_NUMBER + '_' + '_' +
            # '{:02d}'.format(nvir[0]) + '-' +
            ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5')

    if verbose:
        print ('reading from ', lfTotalFileName)

    with h5py.File(lfTotalFileName, 'r') as fileData:

        NBinsM200 = fileData['NumberOfClusterMassBins'][()]


    return NBinsM200

def read_datafile(haloDir, MAG_TYPE, OUTPUT_NUMBER, nvir, DATA_PATH):
    """
    Read everything from a well documented file.

    """

    lfTotalFileName = os.path.join(DATA_PATH ,haloDir , 'dataTotalLF_' + MAG_TYPE +
            OUTPUT_NUMBER + '_' + '_' +
            # '{:02d}'.format(nvir[0]) + '-' +
            ('{0:1.1f}'.format(nvir[0])).replace(".","p") + '-' +
            ('{0:1.1f}'.format(nvir[1])).replace(".","p") + '.h5')

    if verbose:
        print ('reading from ', lfTotalFileName)

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
        norm        = fileData['total/norm'][()]

        y_b      = fileData['starForming/y'][()]
        x_b      = fileData['starForming/x'][()]
        ey_b     = fileData['starForming/ey'][()]
        ex_b     = fileData['starForming/ex'][()]
        numGalaxies_b  = fileData['starForming/numGalaxies'][()]
        mag_min_fit_b = fileData['starForming/mag_min_fit'][()]
        norm_b        = fileData['starForming/norm'][()]

        y_r      = fileData['quiescent/y'][()]
        x_r      = fileData['quiescent/x'][()]
        ey_r     = fileData['quiescent/ey'][()]
        ex_r     = fileData['quiescent/ex'][()]
        numGalaxies_r = fileData['quiescent/numGalaxies'][()]
        mag_min_fit_r = fileData['quiescent/mag_min_fit'][()]
        norm_r        = fileData['quiescent/norm'][()]

    return (y, x, ey, ex, numGalaxies ,
            y_r, x_r, ey_r, ex_r, numGalaxies_r,
            y_b, x_b, ey_b, ex_b, numGalaxies_b, Apertures, zSnap, NBinsM200,
            mag_min_fit, mag_min_fit_r, mag_min_fit_b, norm, norm_r, norm_b)



# ================================================================================
def put_units(tab):
    """
    Put units in the table.

    """

    tab['Mr'].unit = 'mag'
    tab['Mr_p'].unit = 'mag'
    tab['Mr_m'].unit = 'mag'

    tab['Mr_MF_b'].unit = 'mag'
    tab['Mr_MF_b_p'].unit = 'mag'
    tab['Mr_MF_b_m'].unit = 'mag'

    tab['Mr_b'].unit = 'mag'
    tab['Mr_b_p'].unit = 'mag'
    tab['Mr_b_m'].unit = 'mag'

    tab['Mr_f'].unit = 'mag'
    tab['Mr_f_p'].unit = 'mag'
    tab['Mr_f_m'].unit = 'mag'

    tab['Mr_f'].unit = 'mag'
    tab['Mr_f_p'].unit = 'mag'
    tab['Mr_f_m'].unit = 'mag'

    tab['phi'].unit =   'h^3 cMpc^-3 mag'
    tab['phi_p'].unit = 'h^3 cMpc^-3 mag'
    tab['phi_m'].unit = 'h^3 cMpc^-3 mag'

    tab['phi_b'].unit =   'h^3 cMpc^-3 mag'
    tab['phi_b_p'].unit = 'h^3 cMpc^-3 mag'
    tab['phi_b_m'].unit = 'h^3 cMpc^-3 mag'

    tab['phi_f'].unit =   'h^3 cMpc^-3 mag'
    tab['phi_f_p'].unit = 'h^3 cMpc^-3 mag'
    tab['phi_f_m'].unit = 'h^3 cMpc^-3 mag'

    tab['phi_MF_b'].unit =   'h^3 cMpc^-3 mag'
    tab['phi_MF_b_p'].unit = 'h^3 cMpc^-3 mag'
    tab['phi_MF_b_m'].unit = 'h^3 cMpc^-3 mag'

    tab['phi_MF_f'].unit =   'h^3 cMpc^-3 mag'
    tab['phi_MF_f_p'].unit = 'h^3 cMpc^-3 mag'
    tab['phi_MF_f_m'].unit = 'h^3 cMpc^-3 mag'


class color:
       PURPLE = '\033[35m'
       CYAN = '\033[36m'
       BLUE = '\033[34m'
       GREEN = '\033[32m'
       YELLOW = '\033[33m'
       RED = '\033[31m'
       BOLD = '\033[1m'
       UNDERLINE = '\033[4m'
       END = '\033[0m'

# ================================================================================

if __name__ == '__main__':
    import argparse
    r200IntervalsOriginal = [
            [0,0.5],  # 0
            [0,1],    # 1
            [0,2],    # 2
            [0,3],    # 3
            [1,2],    # 4
            [1,3],    # 5
            [1,5],    # 6
            [3,5],    # 7
            [5,10],   # 8
            [0,10],   # 9
            [2,3],    # 10
            [3,4],    # 11
            [4,5],    # 12
            [6,7],    # 13
            [7,9],    # 14
            [1,10],   # 15
            [5,6],    # 16
            [7,8],    # 17
            [8,9],    # 18
            [9,10],   # 19
            [0.5, 1], # 20
            [0.5, 0.8], # 21
            [0.8, 1.3], # 22
            [1.3, 5.],  # 23
            [0, 5.],    # 24
            [1, 5.],    # 25
            [0, 1.4]    # 26
            ]

    from lum_functions.r200_intervals import r200IntervalsOriginal

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
            help='path [%(default)s]', default='.')
    parser.add_argument('-v','--verbose',
            help='verbose output',
            action='store_true')

    parser.add_argument('-o','--overwrite',
            help='over write existing data',
            action='store_true')
    # parser.add_argument('-m', type=str,
            # help='machine [%(default)s]', choices=['virgo','deimos',
            # 'deimos_onlyStars', 'virus', 'macAndrea'], nargs=1,
            # default='deimos')
    parser.add_argument('-r','--radius', nargs='+',
            help='index of the radius bin', type=int,
            choices=set(range(len(r200IntervalsOriginal))))
    parser.add_argument('-S', type=int,
            help='snapshot to process [1-30]', 
            default=list(range(1,30)), choices = range(30), nargs='+')
    args = parser.parse_args()

    verbose = args.verbose

    mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif', size=20)
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.minor.size'] = 5
    plt.ioff()


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

    # # standard ones
    # output_number = [
            # '029_z000p000',
            # '028_z000p036',
            # '027_z000p073',
            # '026_z000p101',
            # '024_z000p155',
            # '022_z000p247',
            # '020_z000p352',
            # '017_z000p474',
            # '014_z000p703',
            # '011_z001p017',
            # '008_z001p493',
            # '006_z001p993',
            # '004_z002p825',
            # # '003_z003p512',
            # # '001_z006p772'
            # ]

    # ALL OF THEM!
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

    if args.radius is not None:
        r200Intervals = [r200IntervalsOriginal[i] for i in args.radius]
    else:
        r200Intervals = r200IntervalsOriginal


    # the target for the descendants, used to get the right directory
    zTarget = 24
    # from helperspy import path_CEAGLE_virgo
    # tmp1, tmp2, listsnapshot_suffixes = path_CEAGLE_virgo(machine=args.m)
    # zTarget = listsnapshot_suffixes[zTarget]
    zTarget = output_number_all [zTarget]
    # del tmp1, tmp2

    LUM_FUNC_PATH = args.path
    lum_func_data_path = 'lum_function_data'
    outputPath  = LUM_FUNC_PATH

    useDensities = True


    main(data, LUM_FUNC_PATH, output_number,
            r200Intervals,
            lum_func_data_path = lum_func_data_path, overwrite=args.overwrite,
            # plot_fit_binned = True
            useDensities = useDensities ,
            #
            # multiplicative constant for all the phi quantities and associated
            # errors in the latex table, useful when the fits are similar to
            # field values, where th value 1000 should be used for pretty
            # results in the latex file
            phiConst = 1000.
            )
