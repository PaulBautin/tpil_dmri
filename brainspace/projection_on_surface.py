
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import glob
import numpy as np
from functions.load_data import load_matrices, load_surface
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

from brainstat.datasets.base import fetch_template_surface, fetch_mask
from brainstat.mesh.interpolate import load_mesh_labels
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM



from neuromaps import datasets, transforms, plotting
pd.options.display.width = 0
# Load image
path_img = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz"
label_lh, label_rh = transforms.mni152_to_fsaverage(path_img, fsavg_density='41k', method='linear')
plotting.plot_surf_template((label_lh, label_rh), template='fsaverage', density='41k')
plt.show()


surf_lh, surf_rh = fetch_template_surface(template="fslr32k", join=False)
mask = fetch_mask(template="fslr32k")
surf_lh, surf_rh = load_surface()
label_lh = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii")
label_rh = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii")
atlas = np.concatenate((label_lh.agg_data(), label_rh.agg_data())).astype("float")
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

