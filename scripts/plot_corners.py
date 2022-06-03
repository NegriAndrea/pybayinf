#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', size=11)
import matplotlib.pylab as plt
from pathlib import PurePath, Path
import h5py
import corner


plt.ioff()
# ================================================================================
def main(chainFilename, outputPath, suffix='', overwrite = False, bay12 = 1,
        verbose=False):

    img_fmt = '.pdf'

    if bay12 == 1:
        gr = '1sc_den'
        labels = [r'$\alpha$', r'$\phi$', r'$M_\star$']
    elif bay12 == 2:
        gr = '2scMoF_fixed_den'
        labels = [r'$\alpha$', r'$\phi$', r'$M_\star$',
                r'$\alpha_\mathrm{f}$', r'$\phi_\mathrm{f}$', r'$M_{\star,f}$']
        labels = [r'$\alpha$', r'$\phi$', r'$M_\star$',
                r'$\alpha_\mathrm{f}$', r'$\phi_\mathrm{f}$',
                r'$\mu$', r'$\sigma$', r'$\phi_\mathrm{g}$']
    else:
        raise IOError('bay12 = ', bay12, ' not valid')


    plotFilename = Path(outputPath) / ('corner_' +
            gr + suffix + img_fmt)

    if plotFilename.is_file() and not overwrite:
        raise IOError(str(plotFilename) + ' is already present')

    flat_chain, chain = read_datafile(chainFilename, gr, verbose=verbose)
    fig = corner.corner(flat_chain, labels=labels,
            show_titles=True,
            levels=1.0 - np.exp(-0.5 * np.array([0.5, 1., 2.]) ** 2))

    if verbose:
        print('saving ', plotFilename)

    fig.savefig(plotFilename)
    plt.close(fig)

    # plot the full chain
    plotFilename = Path(outputPath) / ('chain_' +
            gr+suffix + img_fmt)

    if plotFilename.is_file() and not overwrite:
        raise IOError(str(plotFilename) + ' is already present')

    ndim = chain.shape[-1]
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(labels[i])
        # ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    if verbose:
        print('saving ', plotFilename)

    fig.savefig(plotFilename)
    plt.close(fig)
        


def read_datafile(fileName, groupname, verbose=False):
    """
    Read everything from a well documented file.

    """

    if verbose:
        print('reading from ', fileName)

    with h5py.File(fileName, 'r') as fileData:
        group = fileData[groupname]
        nburninDiscard = group.attrs['nburninDiscard']
        nsteps = group.attrs['nsteps']
        chain = group['samples'][()]
        nwalkers = group.attrs['nwalkers'][()]
        ndim = group.attrs['ndim'][()]

    # cut the chain and flatten it
    flat_cut_chain = np.vstack(chain[nburninDiscard:, :, :])
    assert flat_cut_chain.shape == ((nsteps-nburninDiscard)*nwalkers, ndim)

    return flat_cut_chain, chain


# ================================================================================


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-v','--verbose',
            help='verbose output',
            action='store_true')

    parser.add_argument('chainFile',
            help='hdf5 file containing the chains', type=str)

    parser.add_argument('fitType', type=int,
            help='Type of the fit: 1 single Schecter, 2 double Schecther',
            default=1, choices=[1,2])

    parser.add_argument('-s',
            help='suffix for output files [%(default)s]', type=str, default='')

    parser.add_argument('-o','--overwrite',
            help='over write existing data',
            action='store_true')

    parser.add_argument('--output-path',
            help='output path [%(default)s]',
            default='.')

    args = parser.parse_args()

    plt.ioff()

    main(args.chainFile, args.output_path, overwrite=args.overwrite,
            bay12 = args.fitType, suffix=args.s, verbose=args.verbose)
