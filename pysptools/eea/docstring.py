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
# docstring.py - This file is part of the PySptools package.
#

plot_docstring = """
        Plot the endmembers.

        Parameters:
            path: `string`
              The path where to put the plot.

            axes: `dictionary [default None]`
                * axes['wavelength'] : a wavelengths list (1D python list).
                  If None or not specified the list is automaticaly numbered starting at 1.
                * axes['x'] : the x axis label, 'Wavelength' if None or not specified.
                  axes['x'] is copied verbatim.
                * axes['y'] : the y axis label, 'Brightness' if None or not specified.
                  axes['y'] is copied verbatim.

            suffix: `string [default None]`
                Suffix to add to the file name.
        """

display_docstring = """
        Display the endmembers to a IPython Notebook.

        Parameters:
            axes: `dictionary [default None]`
                * axes['wavelength'] : a wavelengths list (1D python list).
                  If None or not specified the list is automaticaly numbered starting at 1.
                * axes['x'] : the x axis label, 'Wavelength' if None or not specified.
                  axes['x'] is copied verbatim.
                * axes['y'] : the y axis label, 'Brightness' if None or not specified.
                  axes['y'] is copied verbatim.

            suffix: `string [default None]`
                Suffix to add to the title.
        """
