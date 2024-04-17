
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import glob
import numpy as np
from functions.tpil_load_data import load_matrices, load_surface
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

from brainstat.datasets.base import fetch_template_surface, fetch_mask
from brainstat.mesh.interpolate import load_mesh_labels
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM

pd.options.display.width = 0

# Load demographics
df_demographics = pd.read_csv("/home/pabaua/Documents/tpil/CLBP_demographics.csv").drop(columns='Unnamed: 0')
df_demographics = df_demographics.rename(columns={'SubjID (subjectID)': 'subject', 'Age (in years)': 'AGE_AT_SCAN', 'Dx (diagnosis, 0=controls, 1=patients)': 'Dx', 'Sex (0=males, 1=females)': 'SEX'})
fun = lambda x: 'sub-' + str(x.lower().split('pt_')[1])
df_demographics['subject'] = df_demographics['subject'].apply(fun)

### Load connectivity matrices
filter = np.load("/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")
list_clbp = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/clbp/**/Compute_Connectivity/sc.npy')
list_control = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/control/**/Compute_Connectivity/sc.npy')
df_connectivity = load_matrices(list_clbp, list_control)
df_connectivity = df_demographics.merge(df_connectivity)

surf_lh, surf_rh = fetch_template_surface(template="fslr32k", join=False)
mask = fetch_mask(template="fslr32k")
surf_lh, surf_rh = load_surface()
label_lh = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii")
label_rh = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii")
atlas = np.concatenate((label_lhc, label_rh.agg_data())).astype("float")
atlas[atlas == -1] = np.nan
atlas[atlas == 0] = np.nan


mean_conn = np.mean(np.dstack(df_connectivity['connectivity'].values), axis=0).T
map = map_to_labels(mean_conn, atlas, mask=atlas>0, fill=np.nan)
map[np.isnan(map)] = 0
print(np.any(np.isnan(map)))

model = FixedEffect(df_connectivity.AGE_AT_SCAN)
contrast_age = df_connectivity.AGE_AT_SCAN
slm_age = SLM(
    model,
    contrast_age,
    surf="fslr32k",
    mask=mask,
    correction=["fdr", "rft"],
    cluster_threshold=0.01)
slm_age.fit(map)
#skwn, krts = slm_age.qc(map, v=87)

print((mask * 1).shape)

plot_hemispheres(surf_lh, surf_rh, mask * 1, color_bar=True,
        label_text=["mean"], cmap="viridis", size=(1400, 200), zoom=1.45,
        nan_color=(0.7, 0.7, 0.7, 1), interactive=False)

