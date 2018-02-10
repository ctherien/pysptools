#
#------------------------------------------------------------------------------
# Copyright (c) 2013-2017, Christian Therien
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


plot_fi_docstring = """
        Plot the feature importances.
        The output can be split in n graphs.

        Parameters:
            path: `string`
              The path where to save the plot.

            n_labels: `string or integer`
              The number of labels to output by graph. If the value is 'all',
              only one graph is generated.

            height: `float [default 0.2]`
              The bar height (in fact width).

            sort: `boolean [default False]`
              If true the feature importances are sorted.

            suffix: `string [default None]`
              Add a suffix to the file name.
        """


display_fi_docstring = """
        Display the feature importances.
        The output can be split in n graphs.

        Parameters:
            n_labels: `string or integer`
              The number of labels to output by graph. If the value is 'all',
              only one graph is generated.

            height: `float [default 0.2]`
              The bar height (in fact width).

            sort: `boolean [default False]`
              If true the feature importances are sorted.

            suffix: `string [default None]`
              Add a suffix to the file name.
        """
        
