
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


### Load surfaces
# fsaverage midthikness
#surf_lh = mesh_io.read_surface("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.midthickness.32k_fs_LR.surf.gii", itype='gii')
#surf_rh = mesh_io.read_surface("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.midthickness.32k_fs_LR.surf.gii", itype='gii')
# Load pial
surf_lh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/sub-pl007_ses-v1/surf/lh.pial', itype='fs')
surf_rh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/sub-pl007_ses-v1/surf/rh.pial', itype='fs')


# Load label file
annot_lh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/sub-pl007_ses-v1/label/lh.BN_Atlas.annot"
annot_rh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/sub-pl007_ses-v1/label/rh.BN_Atlas.annot"
label = np.concatenate((nib.freesurfer.read_annot(annot_lh)[0], nib.freesurfer.read_annot(annot_rh)[0]), axis=0)
print(np.unique(label))
#label = np.concatenate((nib.load.read_annot(annot_lh)[0], nib.freesurfer.read_annot(annot_rh)[0]), axis=0)

# Create a vector of zeros
#Val = np.repeat(0, annot_lh.n_points + annot_rh.n_points, axis=0)
# Surface color
#grey = plt.colors.ListedColormap(np.full((256, 4), [0.65, 0.65, 0.65, 1]))



#plot_hemispheres(surf_lh, surf_rh, label, size=(800, 200))
#plt.show()

connectivity = np.load("/home/pabaua/dev_tpil/results/results_connectflow/22-10-05_connectflow/results_conn/sub-pl007_ses-v1/Compute_Connectivity/sc_parcel_vol_normalized.npy")
#labels_c = np.loadtxt("/home/pabaua/dev_scil/freesurfer_flow/FS_BN_GL_SF_utils/freesurfer_utils/atlas_brainnetome_v4_labels_list.txt")
values = map_to_labels(connectivity[223], label, fill=np.nan)
plot_hemispheres(surf_lh, surf_rh, values, size=(800, 200), cmap='viridis_r', color_bar=True)

# Reduce matrix size, only for visualization purposes
mat_mask = np.where(np.mean(connectivity, axis=1) > 0.01)[0]
c = connectivity[mat_mask][:, mat_mask]

corr_plot = plotting.plot_matrix(c, figure=(15, 15))
plt.show()


gm = GradientMaps(n_components=2, random_state=0)
gm.fit(connectivity)

