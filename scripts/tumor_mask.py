import os
import numpy as np
import nibabel as nib
from bids import BIDSLayout

"""
# Load data
s1_pre = nib.load('/home/pabaua/david_fortin_dev/data/t1_data_tracto/1_pre_t1.nii.gz')
tumor_mask = nib.load('/home/pabaua/Documents/1_pre_t1_Segmentation-tumor-label.nii.gz')

# remove mask from t1 image
data_1_pre = s1_pre.get_fdata()
data_tumor_mask = tumor_mask.get_fdata()
data_mask = data_1_pre * data_tumor_mask
data_1_pre = data_1_pre - data_mask
data_1_pre = data_1_pre + 0 * data_tumor_mask
img_1_pre_masked = nib.Nifti1Image(data_1_pre, s1_pre.affine)
nib.save(img_1_pre_masked, '/home/pabaua/Documents/1_pre_t1_masked.nii.gz')

# Change binary mask intensity
data_tumor_mask = 0 * tumor_mask.get_fdata()
img_tumor_mask = nib.Nifti1Image(data_tumor_mask, tumor_mask.affine)
nib.save(img_tumor_mask, '/home/pabaua/Documents/1_pre_t1_Segmentation-tumor-label_modified.nii.gz')
"""
#print(BIDSLayout('/home/pabaua/tpil_dev/data/data_dmri_bids_test/test', index_metadata=False))


atlas = nib.load('/home/pabaua/tpil_dev/github/tpil_dmri/to_reference_Warped.nii.gz')
curr_type = atlas.get_data_dtype()
print(curr_type)