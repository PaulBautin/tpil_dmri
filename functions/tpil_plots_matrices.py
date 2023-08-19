import numpy as np
import glob
import nibabel as nib

from brainspace.mesh import mesh_io
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.utils.parcellation import map_to_labels

from neuromaps.parcellate import Parcellater
from neuromaps import datasets, transforms
from neuromaps import datasets
from neuromaps.plotting import plot_surf_template
from neuromaps import stats

from bct import degree, centrality, distance, clustering

from scipy.spatial.distance import squareform, pdist, cdist

from netneurotools.utils import get_centroids
from netneurotools.freesurfer import find_parcel_centroids, parcels_to_vertices
from netneurotools.plotting import plot_fsaverage
from netneurotools.networks import struct_consensus
import netneurotools

import neuromaps

import bct

from mayavi import mlab
from nilearn import plotting
import matplotlib.pyplot as plt
from sklearn.utils.extmath import density

from scipy.stats import zscore

from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.plotting import plot_subcortical


def plot_matrices():
    matrices_clbp = load_matrices(matrices="/home/pabaua/dev_tpil/data/22-11-16_connectflow/clbp/**/Compute_Connectivity/commit2_weights.npy",
                                               filter="/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")
    matrix = np.mean(matrices_clbp, axis=2)
    coordinates = plotting.find_parcellation_cut_coords(labels_img=nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-2mm.nii.gz"))
    plotting.plot_connectome(matrix, coordinates,
                             edge_threshold="80%",
                             title='Yeo Atlas 17 thick (func)')
    plt.show()
    plotting.plot_matrix(matrix)
    plt.show()

# plot_matrices()
def plot_matrix(matrices_g1, matrices_g2):
    pval = matrix_pvalue(matrices_g1, matrices_g2)
    pval[pval > 0.005] = 'nan'
    labels = np.where(~np.isnan(pval).all(axis=0))[0]
    pval = pval[labels, :]
    pval = pval[:, labels]
    label_nb = labels + 1
    df = pd.read_csv("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/BN_Atlas_246_LUT.txt", sep=" ")
    labels = [str(a) + ' : ' + b for a, b in zip(label_nb, df.iloc[labels]['Unknown'].values)]
    plotting.plot_matrix(pval, labels=labels, figure=(9, 7), colorbar=True, cmap='viridis', tri='full')
    plt.show()