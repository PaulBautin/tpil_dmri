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

def connection_strength(m, fig=False):
    m[m == 0] = np.nan
    m[-1:, :] = np.nan
    m[:, -1:] = np.nan
    m = zscore(m, axis=1, nan_policy='omit')
    m_parc = np.nansum(m, axis=0)
    dist_parc_lh, dist_parc_rh = parc.fit().inverse_transform(m_parc)
    dist = np.concatenate((dist_parc_lh.agg_data(), dist_parc_rh.agg_data())).astype("float")
    if fig:
        plotting.plot_matrix(m)
        plt.show()
        plot_hemispheres(surf_lh=surf_lh, surf_rh=surf_rh, array_name=dist, size=(1200, 700), cmap='viridis_r',
                         color_bar=True,
                         nan_color=(0.5, 0.5, 0.5, 0.8),
                         label_text=['Atlas'],
                         zoom=1.5)
    return dist, m_parc
def get_centroids(surfaces, labels):
    surf_lh = nib.load(surfaces['midthickness'].L).agg_data()[0]
    surf_rh = nib.load(surfaces['midthickness'].R).agg_data()[0]
    surfaces = np.concatenate((surf_lh, surf_rh))
    centroids = []
    for lab in np.unique(labels):
        roi = np.atleast_2d(surfaces[labels == lab].mean(axis=0))
        roi = surfaces[np.argmin(cdist(surfaces, roi), axis=0)[0]]
        centroids.append(roi)
    return np.array(centroids)


def load_matrices(matrices, filter):
    #filter = np.load(filter)
    list= glob.glob(matrices)
    matrices = np.dstack(np.array([np.load(path) for path in list]))[:-3,:-3,:]
    return matrices

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

fs = datasets.fetch_atlas(atlas='fsLR', density='164k')
surf_lh = mesh_io.read_surface(fs['midthickness'].L, itype='gii')
surf_rh = mesh_io.read_surface(fs['midthickness'].R, itype='gii')
label_lh = neuromaps.images.load_gifti("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR164k/fsaverage.L.BN_Atlas.164k_fs_LR.label.gii")
label_rh = neuromaps.images.load_gifti("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR164k/fsaverage.R.BN_Atlas.164k_fs_LR.label.gii")
label_lh, label_rh = neuromaps.transforms.fslr_to_fsaverage((label_lh, label_rh),target_density='10k')
print(label_lh.agg_data().shape)
#label_lh.agg_data()[label_lh.agg_data() == -1] = 0.
#label_rh.agg_data()[label_rh.agg_data() == -1] = 0.
#parc = neuromaps.parcellate.Parcellater((label_lh, label_rh), 'fsaverage')
atlas = np.concatenate((label_lh.agg_data(), label_rh.agg_data())).astype("float")

plot_cortical(array_name=atlas, surface_name="fsa5", size=(800, 400),cmap='RdBu_r', color_bar=True, color_range=(-0.5, 0.5))

centroids = get_centroids(fs, atlas[1:])



# compute the euclidian distance between each centroids position
#eu_distance =



eu_distance = squareform(pdist(centroids, metric="euclidean"))
print(eu_distance)
dist_parc_lh, dist_parc_rh = parc.fit().inverse_transform(np.mean(eu_distance, axis=0))
dist = np.concatenate((dist_parc_lh.agg_data(), dist_parc_rh.agg_data())).astype("float")

matrices_clbp = load_matrices(matrices="/home/pabaua/dev_tpil/data/22-11-16_connectflow/clbp/**/Compute_Connectivity/commit2_weights.npy",
                                           filter="/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")
matrices_control = load_matrices(matrices="/home/pabaua/dev_tpil/data/22-11-16_connectflow/control/**/Compute_Connectivity/commit2_weights.npy",
                                           filter="/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")

hemiid = np.array([np.arange(1, 211) % 2 == 0]).T
m_clbp = struct_consensus(matrices_clbp[:210, :210, :], distance=eu_distance, hemiid=hemiid, weighted=True)
m_control = struct_consensus(matrices_control[:210, :210, :], distance=eu_distance, hemiid=hemiid, weighted=True)
print(m_control.shape)

dist_control, m_parc_control = connection_strength(m_control)
dist_clbp, m_parc_clbp = connection_strength(m_clbp)


rotated = neuromaps.nulls.alexander_bloch(m_parc_control, atlas='fsLR', density='164k', n_perm=1000, seed=1234, parcellation=(label_lh, label_rh))
corr, pvalue = stats.compare_images(m_parc_control, m_parc_clbp, nulls=rotated)
print(f'r = {corr:.3f}, p = {pvalue:.3f}')