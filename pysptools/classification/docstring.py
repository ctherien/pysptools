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


classify_docstring = """
        Classify the HSI cube M with the spectral library E.

        Parameters:
            M: `numpy array`
              A HSI cube (m x n x p).

            E: `numpy array`
              A spectral library (N x p).

            threshold: `float [default 0.1] or list`
             * If float, threshold is applied on all the spectra.
             * If a list, individual threshold is applied on each
               spectrum, in this case the list must have the same
               number of threshold values than the number of spectra.
             * Threshold have values between 0.0 and 1.0.

            mask: `numpy array [default None]`
              A binary mask, when *True* the selected pixel is classified.

        Returns: `numpy array`
              A class map (m x n x 1).
        """

get_single_map_docstring = """
        Get individual classified map. See plot_single_map for
        a description.

        Parameters:
            lib_idx: `int or string`
                A number between 1 and the number of spectra in the library.

            constrained: `boolean [default True]`
                See plot_single_map for a description.

        Returns: `numpy array`
            The individual map (m x n x 1) associated to the lib_idx endmember.
        """

plot_single_map_docstring = """
        Plot individual classified map. One for each spectrum.
        Note that each individual map is constrained by the others.
        This function is usefull to see the individual map that compose
        the final class map returned by the classify method. It help
        to define the spectra library. See the constrained parameter below.

        Parameters:
            path: `string`
              The path where to put the plot.

            lib_idx: `int or string`
                * A number between 1 and the number of spectra in the library.
                * 'all', plot all the individual maps.

            constrained: `boolean [default True]`
                * If constrained is True, print the individual maps as they compose the
                  final class map. Any potential intersection is removed in favor of
                  the lower value level for SAM and SID, or the nearest to 1 for NormXCorr. Use
                  this one to understand the final class map.
                * If constrained is False, print the individual maps without intersection
                  removed, as they are generated. Use this one to have the real match.

            stretch: `boolean [default False]`
                Stretch the map between 0 and 1 giving a good distribution of the
                color map.

            colorMap: `string [default 'spectral']`
              A matplotlib color map.

            suffix: `string [default None]`
              Add a suffix to the file name.
        """

display_single_map_docstring = """
        Display individual classified map to a IPython Notebook. One for each spectrum.
        Note that each individual map is constrained by the others.
        This function is usefull to see the individual map that compose
        the final class map returned by the classify method. It help
        to define the spectra library. See the constrained parameter below.

        Parameters:
            lib_idx: `int or string`
                * A number between 1 and the number of spectra in the library.
                * 'all', plot all the individual maps.

            constrained: `boolean [default True]`
                * If constrained is True, print the individual maps as they compose the
                  final class map. Any potential intersection is removed in favor of
                  the lower value level for SAM and SID, or the nearest to 1 for NormXCorr. Use
                  this one to understand the final class map.
                * If constrained is False, print the individual maps without intersection
                  removed, as they are generated. Use this one to have the real match.

            stretch: `boolean [default False]`
                Stretch the map between 0 and 1 giving a good distribution of the
                color map.

            colorMap: `string [default 'spectral']`
              A matplotlib color map.

            suffix: `string [default None]`
              Add a suffix to the title.
        """

plot_docstring = """
        Plot the class map.

        Parameters:
            path: `string`
              The path where to put the plot.

            labels: `list of string [default None]`
                The legend labels. Can be used only if the input spectral library
                E have more than 1 pixel.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding pixel is displayed.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the file name.
        """

display_docstring = """
        Display the class map to a IPython Notebook.

        Parameters:
            labels: `list of string [default None]`
                The legend labels. Can be used only if the input spectral library
                E have more than 1 pixel.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding pixel is displayed.

            interpolation: `string [default none]`
              A matplotlib interpolation method.

            colorMap: `string [default 'Accent']`
              A color map element of
              ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3'],
              "Accent" is the default and it fall back on "Jet".

            suffix: `string [default None]`
              Add a suffix to the title.
        """

plot_histo_docstring = """
        Plot the histogram.

        Parameters:
            path: `string`
              The path where to put the plot.

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
