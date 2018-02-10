"""
Plot abundance maps stack for the methanol gas HSI cube.
"""

from __future__ import print_function

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp


def parse_ENVI_header(head):
    ax = {}
    ax['wavelength'] = head['wavelength']
    ax['x'] = 'Wavelength - '+head['z plot titles'][0]
    ax['y'] = head['z plot titles'][1]
    return ax


def get_endmembers(data, header, result_path):
    print('Endmembers extraction with NFINDR')
    nfindr = eea.NFINDR()
    U = nfindr.extract(data, 12, maxit=5, normalize=True, ATGP_init=True)
    nfindr.plot(result_path, axes=header, suffix='gas')
    # return an array of endmembers
    return U


def gen_abundance_maps(data, U, result_path):
    print('Abundance maps generation with NNLS')
    nnls = amp.NNLS()
    amaps = nnls.map(data, U, normalize=True)
    nnls.plot(result_path, colorMap='jet', suffix='gas')
    # return an array of abundance maps
    return amaps


def plot_synthetic_gas_above(amap, colormap, result_path):
    print('Create and plot synthetic map for the gas_above')
    gas = (amap > 0.1) * amap
    stack = gas[:,:,4] + gas[:,:,5] + gas[:,:,9] + gas[:,:,10]
    plot_synthetic_image(stack, colormap, 'gas_above', result_path)


def plot_synthetic_gas_around(amap, colormap, result_path):
    print('Create and plot synthetic map for the gas_around')
    gas = (amap > 0.1) * amap
    stack = gas[:,:,6] + gas[:,:,7] + gas[:,:,8]
    plot_synthetic_image(stack, colormap, 'gas_around', result_path)


def plot_synthetic_burner(amap, colormap, result_path):
    print('Create and plot synthetic map for the burner')
    burner = (amap > 0.1) * amap
    stack = burner[:,:,2] + burner[:,:,3]
    plot_synthetic_image(stack, colormap, 'burner', result_path)


def plot_synthetic_image(image, colormap, desc, result_path):
    plt.ioff()
    img = plt.imshow(image, interpolation='none')
    img.set_cmap(colormap)
    plt.colorbar()
    fout = osp.join(result_path, 'synthetic_{0}.png'.format(desc))
    plt.savefig(fout)
    plt.clf()


if __name__ == '__main__':
    # Load the cube
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    
    sample = 'burner.hdr'
    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)

    result_path = osp.join(home, 'results')

    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    axes = parse_ENVI_header(header)

    # Telops cubes are flipped left-right
    # Flipping them again restore the orientation
    data = np.fliplr(data)

    U = get_endmembers(data, axes, result_path)
    amaps = gen_abundance_maps(data, U, result_path)
    # best color maps: hot and spectral
    plot_synthetic_gas_above(amaps, 'spectral', result_path)
    plot_synthetic_gas_around(amaps, 'hot', result_path)
    plot_synthetic_burner(amaps, 'hot', result_path)
