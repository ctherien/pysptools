"""
Plot smokestack effluents.
"""

from __future__ import print_function

import os
import os.path as osp
import pysptools.classification as cls
import matplotlib.pyplot as plt
import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp

import numpy as np

n_emembers = 8


def parse_ENVI_header(head):
    ax = {}
    ax['wavelength'] = head['wavelength']
    ax['x'] = 'Wavelength - '+head['z plot titles'][0]
    ax['y'] = head['z plot titles'][1]
    return ax


class Classify(object):
    """
    For this problem NormXCorr works as well as SAM
    SID was not tested.
    """

    def __init__(self, data, E, path, threshold, suffix):
        print('Classify using SAM')
        self.sam = cls.SAM()
        self.sam.classify(data, E, threshold=threshold)
        self.path = path
        self.suffix = suffix

    def get_single_map(self, idx):
        return self.sam.get_single_map(idx, constrained=False)

    def plot_single_map(self, idx):
        self.sam.plot_single_map(self.path, idx, constrained=False, suffix=self.suffix)

    def plot(self):
        self.sam.plot(self.path, suffix=self.suffix)


def get_endmembers(data, header, q, path, mask, suffix, output=False):
    print('Endmembers extraction with NFINDR')
    ee = eea.NFINDR()
    U = ee.extract(data, q, maxit=5, normalize=True, ATGP_init=True, mask=mask)
    if output == True:
        ee.plot(path, axes=header, suffix=suffix)
    return U


def get_abundance_maps(data, U, umix_source, path, output=False):
    print('Abundance maps with FCLS')
    fcls = amp.FCLS()
    amap = fcls.map(data, U, normalize=True)
    if output == True:
        fcls.plot(path, colorMap='jet', suffix=umix_source)
    return amap


def get_full_cube_em_set(data, header, path):
    """ Return a endmembers set for the full cube and a region of interest (ROI).
        The ROI is created using a small region of the
        effluents leaving near the smokestack.
    """
    # Take the endmembers set for all the cube
    U = get_endmembers(data, header, n_emembers, path, None, 'full_cube', output=True)
    # A threshold of 0.15 give a good ROI
    cls = Classify(data, U, path, 0.15, 'full_cube')
    # The endmember EM2 is use to define the region of interest
    # i.e. the effluents region of interest
    effluents = cls.get_single_map(2)
    # Create the binary mask with the effluents
    mask = (effluents > 0)
    # Plot the mask
    plot(mask, 'gray', 'binary_mask', path)
    return U, mask


def get_masked_em_set(data, header, path, mask):
    """ Return a endmembers set that belong to the ROI (mask).
    """
    # Use the mask to extract endmembers near the smokestack exit
    U = get_endmembers(data, header, n_emembers, path, mask, 'masked', output=True)
    return U


def classification_analysis(data, path, E_masked):
    # Note: the classification is done with NormXCorr instead of SAM
    # Classify with the masked endmembers set
    c = cls.NormXCorr()
    c.classify(data, E_masked, threshold=0.15)
    c.plot_single_map(path, 'all', constrained=False, suffix='masked')
    c.plot(path, suffix='masked')
    # Calculate the average image
    gas = c.get_single_map(1, constrained=False)
    for i in range(n_emembers - 1):
        gas = gas + c.get_single_map(i+2, constrained=False)
    gas = gas / n_emembers
    # and plot it
    plot(gas, 'spectral', 'mean_NormXCorr', path)


def unmixing_analysis(data, path, E_full_cube, E_masked):
    # Calculate an unmixed average image at the ROI position.
    # Each endmember belonging to E_masked takes place inside E_full_cube at
    # the ROI position. Netx, we sum the abundance maps
    # generated at this position. And finally a mean is calculated.
    for i in range(n_emembers):
        E_full_cube[1,:] = E_masked[i,:]
        amaps = get_abundance_maps(data, E_full_cube, 'masqued_{0}'.format(i+1), path, output=False)
        if i == 0:
            mask = amaps[:,:,1]
        else:
            mask = mask + amaps[:,:,1]
        plot(amaps[:,:,1], 'spectral', 'FCLS_masqued_{0}'.format(i+1), path)
    mask = mask / n_emembers
    thresholded = (mask > 0.15) * mask
    plot(thresholded, 'spectral', 'mean_FCLS', path)


def plot(image, colormap, desc, path):
    plt.ioff()
    img = plt.imshow(image, interpolation='none')
    img.set_cmap(colormap)
    plt.colorbar()
    fout = osp.join(path, '{0}.png'.format(desc))
    plt.savefig(fout)
    plt.clf()


if __name__ == '__main__':
    plt.ioff()

    # Load the cube
    data_path = os.environ['PYSPTOOLS_DATA']
    home = os.environ['HOME']
    
    sample = 'Smokestack1.hdr'
    data_file = osp.join(data_path, sample)
    data, header = util.load_ENVI_file(data_file)

    result_path = osp.join(home, 'results')

    if osp.exists(result_path) == False:
        os.makedirs(result_path)

    axes = parse_ENVI_header(header)

    # Telops cubes are flipped left-right
    # Flipping them again restore the orientation
    data = np.fliplr(data)

    U_full_cube, mask = get_full_cube_em_set(data, axes, result_path)
    U_masked = get_masked_em_set(data, axes, result_path, mask)
    classification_analysis(data, result_path, U_masked)
    unmixing_analysis(data, result_path, U_full_cube, U_masked)
