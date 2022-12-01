
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh import mesh_io
from brainspace.datasets import load_group_fc, load_parcellation
from brainspace.gradient import GradientMaps
import numpy as np
from brainspace.utils.parcellation import map_to_labels
from nilearn import plotting
from nilearn import datasets
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import glob
import scipy
import brainspace
from nilearn.connectome import ConnectivityMeasure

from neuromaps import datasets


def load_surface():
    surf_lh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/surf/lh.pial',
                                   itype='fs')
    surf_rh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/surf/rh.pial',
                                   itype='fs')
    return surf_lh, surf_rh
def load_annot():
    # Brainnetome Atlas has 210 cortical and 36 subcortical subregions
    # vertices with no id have an id set to -1 (Example: subcortical regions)
    annot_lh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/label/lh.BN_Atlas.annot"
    annot_rh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/label/rh.BN_Atlas.annot"
    atlas = np.concatenate((nib.freesurfer.read_annot(annot_lh)[0], nib.freesurfer.read_annot(annot_rh)[0]),
                           axis=0).astype(float)
    atlas[atlas <= 0] = np.nan
    return atlas
def load_matrices():
    ### Load connectivity matrices
    filter = np.load("/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")
    list_g1 = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/clbp/**/Compute_Connectivity/sc.npy')
    list_g2 = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/control/**/Compute_Connectivity/sc.npy')
    matrices_g1 = np.dstack(np.array([np.load(path) * filter for path in list_g1]))[:-3,:-3,:]
    matrices_g2 = np.dstack(np.array([np.load(path) * filter for path in list_g2]))[:-3,:-3,:]
    # matrices_g1 = np.dstack(np.array([np.load(path) for path in list_g1]))[:-3, :-3, :]
    # matrices_g2 = np.dstack(np.array([np.load(path) for path in list_g2]))[:-3, :-3, :]
    return matrices_g1, matrices_g2


def matrix_pvalue(matrices_g1, matrices_g2):
    pval = scipy.stats.ttest_ind(matrices_g1, matrices_g2, axis=2, nan_policy='propagate')[1]
    return pval


def matrix_mean(matrix):
    mat = np.mean(matrix, axis=2)
    return mat


def degree_count(matrix):
    matrix[matrix > 0] = 1
    mat = np.nansum(matrix, axis=1)
    return mat


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

def plot_conn_to_surf(matrices_g1, matrices_g2, atlas):
    pval = matrix_pvalue(matrices_g1, matrices_g2)
    mean_g1 = matrix_mean(matrices_g1)
    mean_g2 = matrix_mean(matrices_g2)
    pval[pval > 0.05] = 'nan'
    pval[pval <= 0.05] = 1
    print(np.where(pval[222, :] == 1))
    maps = [atlas] + [map_to_labels(g[222, :], atlas, mask=atlas > 0, fill=np.nan) for g in [mean_g1, mean_g2, pval]]
    plot_hemispheres(surf_lh=surf_lh, surf_rh=surf_rh, array_name=maps, size=(1200, 700), cmap='viridis_r',
                     color_bar=True,
                     nan_color=(0.5, 0.5, 0.5, 0.8),
                     label_text=['Atlas', 'CLBP', 'Control', 'p-value < 0.05'],
                     zoom=1.5)

def plot_deg_to_surf(matrices_g1, matrices_g2, atlas):
    pval = matrix_pvalue(matrices_g1, matrices_g2)
    mean_g1 = matrix_mean(matrices_g1)
    mean_g2 = matrix_mean(matrices_g2)
    pval[pval > 0.05] = 'nan'
    pval[pval <= 0.05] = 1
    dg_count = degree_count(mean_g1)
    dg_count_pval = degree_count(pval)
    print(np.where(dg_count_pval >= 4))
    dg_count[dg_count == 0] = 'nan'
    dg_count_pval[dg_count_pval == 0] = 'nan'
    dg_count_norm = dg_count_pval / dg_count
    print(np.where(dg_count_norm >= 0.5))
    maps = [atlas] + [map_to_labels(g, atlas, mask=atlas > 0, fill=np.nan) for g in [dg_count, dg_count_pval, dg_count_norm]]
    plot_hemispheres(surf_lh=surf_lh, surf_rh=surf_rh, array_name=maps, size=(1200, 700), cmap='viridis_r',
                     color_bar=True,
                     nan_color=(0.5, 0.5, 0.5, 0.8),
                     label_text=['Atlas', 'dg_count', 'dg_count_pval', 'dg_count_norm'],
                     zoom=1.5)

def plot_gradient_to_surf(matrices_g1, atlas):
    ### compute stats from connectivity matrices
    gm = GradientMaps(n_components=6)
    mean_g1 = matrix_mean(matrices_g1)
    gm.fit(mean_g1)
    # Map gradients to original parcels
    mask = atlas >= 0
    grad = np.sum(gm.gradients_, axis=1)
    print(grad.shape)
    grad = [map_to_labels(gm.gradients_[:,i], atlas, mask=mask, fill=np.nan) for i in np.arange(6)]
    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 1200), cmap='viridis_r',
                     color_bar=True, label_text=['Grad1', 'Grad2', 'Grad3', 'Grad4', 'Grad5', 'Grad6'], zoom=1.5,
                     nan_color=(0.5, 0.5, 0.5, 0.8))

def plot_gradient_to_surf_2(matrices_g1, matrices_g2, atlas):
    ### compute stats from connectivity matrices
    gm = GradientMaps(n_components=6, alignment='procrustes')
    mean_g1 = matrix_mean(matrices_g1)
    mean_g2 = matrix_mean(matrices_g2)
    gm.fit([mean_g1, mean_g2])
    # Map gradients to original parcels
    mask = atlas >= 0
    grad = [map_to_labels(g[:, 0], atlas, mask=mask, fill=np.nan) for g in gm.aligned_]
    grad += [map_to_labels(g[:, 1], atlas, mask=mask, fill=np.nan) for g in gm.aligned_]
    grad += [map_to_labels(g[:, 2], atlas, mask=mask, fill=np.nan) for g in gm.aligned_]
    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 1200), cmap='viridis_r',
                     color_bar=True, label_text=['Grad1', 'Grad2', 'Grad3'], zoom=1.5)


# To load
surf_lh, surf_rh = load_surface()
atlas = load_annot()
matrices_g1, matrices_g2 = load_matrices()


plot_conn_to_surf(matrices_g1, matrices_g2, atlas)
plot_deg_to_surf(matrices_g1, matrices_g2, atlas)

### plot clonnectivity
plot_matrix(matrices_g1, matrices_g2)

# plot_gradient_to_surf(matrices_g1, atlas)