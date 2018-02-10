#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2014, Christian Therien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------
#
# display.py - This file is part of the PySptools package.
#

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


def plot_linear_stretch(M, path, R, G, B, suffix=None):
    """
    Plot a linear stretched RGB image.

    Parameters:
        M: `numpy array`
          A HSI cube (m x n x p).

        path: `string`
          The path where to put the plot.

        R: `int`
            A band number that will render the red color.

        G: `int`
            A band number that will render the green color.

        B: `int`
            A band number that will render the blue color.

        suffix: `string [default None]`
          Add a suffix to the file name.
    """
    img = _linear_stretch(M, R, G, B)
    plt.ioff()
    if suffix == None:
        fout = osp.join(path, 'linear_stretch.png')
    else:
        fout = osp.join(path, 'linear_stretch_{0}.png'.format(suffix))
    plt.imsave(fout, img)
    plt.close()


def display_linear_stretch(M, R, G, B, suffix=None):
    """
    Display a linear stretched RGB image.

    Parameters:
        M: `numpy array`
          A HSI cube (m x n x p).

        R: `int`
            A band number that will render the red color.

        G: `int`
            A band number that will render the green color.

        B: `int`
            A band number that will render the blue color.

        suffix: `string [default None]`
          Add a suffix to the title.
    """
    img = _linear_stretch(M, R, G, B)
    plt.imshow(img, interpolation='none')
    if suffix == None:
        plt.title('Linear Stretch')
    else:
        plt.title('Linear Stretch - {0}'.format(suffix))
    plt.show()
    plt.close()


def _linear_stretch(data, R, G, B):
    """ Do a linear stretch. """
    img = np.zeros((data.shape[0],data.shape[1],3), dtype=np.float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            img[i,j] = data[i,j,R], data[i,j,G], data[i,j,B]
    d_R = np.max(img[:,:,0])-np.min(img[:,:,0])
    min_R = np.min(img[:,:,0])
    d_G = np.max(img[:,:,1])-np.min(img[:,:,1])
    min_G = np.min(img[:,:,1])
    d_B = np.max(img[:,:,2])-np.min(img[:,:,2])
    min_B = np.min(img[:,:,2])
    img1 = np.zeros((data.shape[0],data.shape[1],3), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            img1[i,j,0] = (1-((img[i,j,0]-min_R)/d_R))*255
            img1[i,j,1] = (1-((img[i,j,1]-min_G)/d_G))*255
            img1[i,j,2] = (1-((img[i,j,2]-min_B)/d_B))*255
    return 255-img1

if __name__ == '__main__':
    import os
    import pysptools.util as util
    home_path = os.environ['HOME']
    source_path = osp.join(home_path, 'dev-data/data/data1_dis')
    result_path = osp.join(home_path, 'results')
    sample = '92AV3C.hdr'

    data_file = osp.join(source_path, sample)
    data, info = util.load_ENVI_file(data_file)

    plot_linear_stretch(data, result_path, 102, 85, 18, '1')
    plot_linear_stretch(data, result_path, 98, 86, 22, '2')
    plot_linear_stretch(data, result_path, 75, 34, 0, '3')
    plot_linear_stretch(data, result_path, 74, 46, 18, '4')
