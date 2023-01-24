import numpy as np
import glob
import pandas as pd
from brainspace.mesh import mesh_io

def load_matrices(list_g1, list_g2, filter=1):
    matrices_g1 = {path.split('clbp/')[1].split('/Compute_Connectivity/')[0]: [np.load(path)[:-3,:-3] * filter] for path in list_g1}
    matrices_g2 = {path.split('control/')[1].split('/Compute_Connectivity/')[0]: [np.load(path)[:-3,:-3] * filter] for path in list_g2}
    df_matrices_g1 = pd.DataFrame.from_dict(matrices_g1, orient='index', columns=['connectivity']).reset_index()
    df_matrices_g2 = pd.DataFrame.from_dict(matrices_g2, orient='index', columns=['connectivity']).reset_index()
    df_matrices_g1[['subject', 'session']] = df_matrices_g1['index'].str.rsplit('_ses-', 1, expand=True)
    df_matrices_g2[['subject', 'session']] = df_matrices_g2['index'].str.rsplit('_ses-', 1, expand=True)
    df = pd.concat([df_matrices_g1.drop(columns='index'), df_matrices_g2.drop(columns='index')])
    return df


def load_surface():
    surf_lh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.midthickness.32k_fs_LR.surf.gii')
    surf_rh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.midthickness.32k_fs_LR.surf.gii')
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





