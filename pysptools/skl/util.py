#

import numpy as np
from sklearn import preprocessing


def _plot_feature_importances(ID, importance, path=None, n_labels='all',
                              height=0.2, sort=False, suffix=None):
    import os.path as osp
    import matplotlib.pyplot as plt

    n_features_ = len(importance)
    nb_bands_pos = np.arange(n_features_)
    bands_id = [str(n+1) for n in range(n_features_)]
    n_graph = 1
    x = 0
    min_ = np.min(importance)
    max_ = np.max(importance)
    bar_height = height

    if sort == True:
        tuples = [(str(k+1), importance[k]) for k in range(n_features_)]
        tuples = sorted(tuples, key=lambda x: x[1])
        tuples.reverse()
        bands_id, importance = zip(*tuples)
    
    if n_labels == 'all':
        nb_to_plot = n_features_
    else:
        nb_to_plot = n_labels
    for i in range(n_features_):
        if (i+1) % nb_to_plot == 0 :
            if path != None: plt.ioff()
            fig, ax = plt.subplots()
            nb_bands_pos = np.arange(nb_to_plot)

            ax.barh(nb_bands_pos, importance[x:i+1],
                    align='center', color='green', height=bar_height)
            ax.set_yticks(nb_bands_pos)
            ax.set_yticklabels(bands_id[x:i+1])
            ax.set_ylim(-1, nb_to_plot)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xbound(min_, max_)
            ax.set_xlabel('Feature Importances Score')
            ax.set_ylabel('band #')
    
            if path != None:
                ax.set_title('Feature Importances #{}'.format(n_graph))
                if suffix == None:
                    fout = osp.join(path, '{0}_feat_imp_{1}.png'.format(ID, n_graph))
                else:
                    fout = osp.join(path, '{0}_feat_imp_{1}_{2}.png'.format(ID, suffix, n_graph))
                try:
                    plt.savefig(fout)
                except IOError:
                    raise IOError('in pysptools.sklearn.util._plot_feature_importances, no such file or directory: {0}'.format(path))
            else:
                if suffix == None:
                    ax.set_title('Feature Importances #{}'.format(n_graph))
                else:
                    ax.set_title('Feature Importances #{0} {1}'.format(n_graph, suffix))
                plt.show()
            n_graph += 1
            x = i+1
            plt.clf()
            plt.close()
    if x < n_features_:
        if path != None: plt.ioff()
        end = n_features_
        fig, ax = plt.subplots()
        nb_bands_pos = np.arange(end-x)
        ax.barh(nb_bands_pos, importance[x:end],
                align='center', color='green', height=bar_height)
        ax.set_yticks(nb_bands_pos)
        ax.set_yticklabels(bands_id[x:end])
        ax.set_ylim(-1, end-x)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xbound(min_, max_)
        ax.set_xlabel('Feature Importances Score')
        ax.set_ylabel('band #')

        if path != None:
            ax.set_title('Feature Importances #{}'.format(n_graph))
            if suffix == None:
                fout = osp.join(path, '{0}_feat_imp_{1}.png'.format(ID, n_graph))
            else:
                fout = osp.join(path, '{0}_feat_imp_{1}_{2}.png'.format(ID, suffix, n_graph))
            try:
                plt.savefig(fout)
            except IOError:
                raise IOError('in pysptools.sklearn.util._plot_feature_importances, no such file or directory: {0}'.format(path))
        else:
            if suffix == None:
                ax.set_title('Feature Importances #{}'.format(n_graph))
            else:
                ax.set_title('Feature Importances #{0} {1}'.format(n_graph, suffix))
            plt.show()
        plt.clf()          
    plt.close()

    
def hyper_scale(M):
    """
    Center a hyperspectral image to the mean and 
    component wise scale to unit variance.
    
    Call scikit-learn preprocessing.scale()
    """
    h, w, numBands = M.shape
    X = np.reshape(M, (w*h, numBands))
    X_scaled = preprocessing.scale(X)
    return np.reshape(X_scaled, (h, w, numBands))


def shape_to_XY(M_list, cmap_list):
    """
    Receive as input a hypercubes list and the corresponding
    masks list. The function reshape and concatenate both to create the X and Y
    arrays.

    Parameters:
        M_list: `numpy array list`
            A list of HSI cube (m x n x p).

        cmap_list: `numpy array list`
            A list of class map (m x n), as usual the classes
            are numbered: 0 for the background, 1 for the first class ...
    """
    def convert2D(M):
        h, w, numBands = M.shape
        return np.reshape(M, (w*h, numBands))

    i = 0
    for m,msk in zip(M_list, cmap_list):
        x = convert2D(m)
        y = np.reshape(msk, msk.shape[0]*msk.shape[1])
        if i == 0:
            X = x
            Y = y
            i = 1
        else:
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))
    return X,Y
