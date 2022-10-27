import os
import numpy as np
import nibabel as nib

def main():
    # Load atlas data
    atlas = nib.load('/home/pabaua/tpil_dev/github/tpil_dmri/warped_atlas_int.nii.gz')
    data_atlas = atlas.get_fdata()
    # Create masks
    # mPFC
    mask_mPFC = (data_atlas == 27) | (data_atlas == 45) | (data_atlas == 47)
    mPFC = nib.Nifti1Image(mask_mPFC.astype(int), atlas.affine)
    nib.save(mPFC, '/home/pabaua/tpil_dev/results/mask_mPFC.nii.gz')

    # NAC
    mask_NAC = (data_atlas == 219) | (data_atlas == 223)
    NAC = nib.Nifti1Image(mask_NAC.astype(int), atlas.affine)
    nib.save(NAC, '/home/pabaua/tpil_dev/results/mask_NAC.nii.gz')

if __name__ == "__main__":
    main()