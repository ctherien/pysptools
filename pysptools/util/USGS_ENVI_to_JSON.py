"""
"""

import os.path as osp
import pysptools.util as util
import numpy as np


def translate_lib_envi(path, fname):
    import json
    data_file = osp.join(path, fname+'.hdr')
    data, info = util.load_ENVI_spec_lib(data_file)
    # process the header file
    info_out = osp.join(path, fname+'.jhead')
    with open(info_out, 'w') as content_file:
        content_file.write(json.dumps(info))
    # process the data file
    fout = osp.join(path, fname+'.jdata')
    with open(fout, 'w+') as content_file:
        content_file.write(json.dumps(data.tolist()))


if __name__ == '__main__':
    translate_lib_envi('../usgs', 's06av95a_envi')
