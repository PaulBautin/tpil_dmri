from brainspace.plotting import plot_hemispheres
from brainspace.mesh import mesh_io
import numpy as np
from brainspace.utils.parcellation import map_to_labels
import nibabel as nib
import glob

from neuromaps import datasets, transforms
from neuromaps import stats
from neuromaps import nulls
from neuromaps.parcellate import Parcellater

import bct

def annot_to_gifti(atlas):
    labels, ctab, names = nib.freesurfer.read_annot(atlas, orig_ids=False)
    labels[labels <= 0] = '0'.encode('UTF-8')
    darr = nib.gifti.GiftiDataArray(labels, intent='NIFTI_INTENT_LABEL',
                                    datatype='NIFTI_TYPE_INT32')
    labeltable = nib.gifti.GiftiLabelTable()
    for key, label in enumerate(names):
        (r, g, b), a = (ctab[key, :3] / 255), (1.0 if key != 0 else 0.0)
        glabel = nib.gifti.GiftiLabel(key, r, g, b, a)
        glabel.label = label.decode()
        labeltable.labels.append(glabel)
    return nib.GiftiImage(darrays=[darr], labeltable=labeltable)


def load_atlas():
    annot_lh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/label/lh.BN_Atlas.annot"
    annot_rh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/label/rh.BN_Atlas.annot"
    parc_lh = annot_to_gifti(annot_lh)
    parc_rh = annot_to_gifti(annot_rh)
    atlas = np.concatenate((parc_lh.agg_data(), parc_rh.agg_data())).astype(float)
    atlas[atlas == 0] = np.nan
    # All regions with an ID of 0 are ignored.
    parc = Parcellater((parc_lh, parc_rh), 'fsaverage')
    return atlas, parc, parc_lh, parc_rh


def load_matrices():
    ### Load connectivity matrices
    filter = np.load("/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")
    list_g1 = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/clbp/**/Compute_Connectivity/sc_edge_normalized.npy')
    list_g2 = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/control/**/Compute_Connectivity/sc_edge_normalized.npy')
    matrices_g1 = np.dstack(np.array([np.load(path) for path in list_g1]))[:-3,:-3,:]
    matrices_g2 = np.dstack(np.array([np.load(path) for path in list_g2]))[:-3,:-3,:]
    return matrices_g1, matrices_g2


fs = datasets.fetch_atlas(atlas='fsaverage', density='164k')
fs_lh = mesh_io.read_surface(fs['pial'].L, itype='gii')
fs_rh = mesh_io.read_surface(fs['pial'].R, itype='gii')
print(fs_lh)
plot_hemispheres(surf_lh=fs_lh, surf_rh=fs_rh, size=(1200, 400), cmap='viridis_r',
                 color_bar=True,
                 nan_color=(0.5, 0.5, 0.5, 0.8),
                 zoom=1.5)


nsynth = datasets.fetch_annotation(source='margulies2016', desc='fcgradient05')
fslr_neuro_lh, fslr_neuro_rh = transforms.fslr_to_fslr(nsynth, '164k')
neuro = np.concatenate((fslr_neuro_lh.agg_data(), fslr_neuro_rh.agg_data()), axis=0)
neuro[neuro == 0] = np.nan

### Load atlas file
atlas, parc, parc_lh, parc_rh = load_atlas()

matrices_g1, matrices_g2 = load_matrices()
matrices = np.dstack((matrices_g1, matrices_g2))

mean = np.mean(matrices_g2, axis=2)
mean[mean > 0] = 1
mean = np.nansum(mean, axis=1)


surf_deg = map_to_labels(mean, atlas, mask=atlas > 0, fill=np.nan)
pval_parc = parc.fit_transform(surf_deg, 'fsaverage')
neuro_parc = parc.fit_transform(neuro, 'fsaverage', ignore_background_data=True)
surf_neuro = map_to_labels(neuro_parc, atlas, mask=atlas > 0, fill=np.nan)

plot_hemispheres(surf_lh=fs_lh, surf_rh=fs_rh, array_name=[atlas, neuro, surf_neuro, surf_deg], size=(1200, 700), cmap='viridis_r', color_bar=True,
                 nan_color=(0.5,0.5,0.5,1), zoom=1.5, label_text=['Atlas', 'Neuro', 'Neuro parc', 'Struct deg'])


rotated = nulls.alexander_bloch(neuro_parc, atlas='fsaverage', density='164k', n_perm=1000, seed=1234, parcellation=(parc_lh, parc_rh))
corr, pvalue = stats.compare_images(neuro_parc, pval_parc, nulls=rotated)
print(f'r = {corr:.3f}, p = {pvalue:.3f}')
