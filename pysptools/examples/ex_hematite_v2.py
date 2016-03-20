"""
Plot a quartz class map for a drill core HSI cube.
"""

from __future__ import print_function

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp


def parse_ENVI_header(head):
    ax = {}
    ax['wavelength'] = head['wavelength']
    ax['x'] = 'Wavelength - '+head['z plot titles'][0]
    ax['y'] = head['z plot titles'][1]
    return ax


def get_endmembers(data, info, q, path):
    print('Endmembers extraction with NFINDR')
    ee = eea.NFINDR()
    U = ee.extract(data, q, maxit=5, normalize=True, ATGP_init=True)
    ee.plot(path, axes=info)
    return U


def gen_abundance_maps(data, U, result_path):
    print('Abundance maps with FCLS')
    fcls = amp.FCLS()
    amap = fcls.map(data, U, normalize=True)
    fcls.plot(result_path, colorMap='jet')
    return amap


def plot(image, colormap, desc, path):
    plt.ioff()
    img = plt.imshow(image, interpolation='none')
    img.set_cmap(colormap)
    plt.colorbar()
    fout = osp.join(path, 'plot_{0}.png'.format(desc))
    plt.savefig(fout)
    plt.clf()


if __name__ == '__main__':
    # Load the cube
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    result_path = os.path.join(home, 'results')

    sample = 'hematite.hdr'
    
    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)

    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    axes = parse_ENVI_header(header)

    # Telops cubes are flipped left-right
    # Flipping them again restore the orientation
    data = np.fliplr(data)

    U = get_endmembers(data, axes, 4, result_path)
    amaps = gen_abundance_maps(data, U, result_path)

    # EM4 == quartz
    quartz = amaps[:,:,3]
    plot(quartz, 'spectral', 'quartz', result_path)

    # EM1 == background, we use the backgroud to isolate the drill core
    # and define the mask
    mask = (amaps[:,:,0] < 0.2)
    plot(mask, 'spectral', 'mask', result_path)

    # Plot the quartz in color and the hematite in gray
    plot(np.logical_and(mask == 1, quartz <= 0.001) + quartz, 'spectral', 'hematite+quartz', result_path)

    # pixels stat
    rock_surface = np.sum(mask)
    quartz_surface = np.sum(quartz > 0.16)
    print('Some statistics')
    print('  Drill core surface (mask) in pixels:', rock_surface)
    print('  Quartz surface in pixels:', quartz_surface)
    print('  Hematite surface in pixels:', rock_surface - quartz_surface)
