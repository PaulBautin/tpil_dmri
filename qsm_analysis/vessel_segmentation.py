#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import skimage
import nibabel as nib
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import frangi, gaussian, meijering, hessian
from skimage.exposure import equalize_adapthist
from skimage.morphology import binary_opening, disk
from skimage.restoration import denoise_nl_means, estimate_sigma

# High resolution vasculature data is challenging to obtain with full-brain imaging techniques. QSM provides an alternative contrast to
# depict venous vessels in high-resolution images, which is not impacted by signal variations due to relaxation parameters or non-local field
# perturbations seen in phase images and SWI. The proposed segmentation method provides a fast segmentation result comparable to manual
# segmentations, a particularly long and tedious task for smaller cortical vessels.

input_image = '/home/pabaua/dev_tpil/results/results_qsm/results_qsm_nextqsm/qsm/sub-002_ses-v2_echo-1_part-phase_MEGRE_scaled_resampled_B0_phase-TE005_tgvqsmjl_resampled_ref.nii'
#input_image = '/home/pabaua/dev_tpil/mask_r2s.nii.gz'
output_image = '/home/pabaua/dev_tpil/sub-002_ses-v2_echo-1_part-phase_MEGRE_scaled_resampled_B0_phase-TE005_tgvqsmjl_resampled_ref_seg.nii.gz'
#nighres.filtering.multiscale_vessel_filter(input_image, structure_intensity='bright', file_name=output_image)


qsm_image = nib.load(input_image).get_fdata()
qsm_image = qsm_image[80:310,50:330,:]





#qsm_image = skimage.transform.rescale(qsm_image, 1.2)
sigma_est = np.mean(estimate_sigma(qsm_image, channel_axis=-1))
print(f'estimated noise standard deviation = {sigma_est}')
patch_kw = dict(patch_size=8,      # 5x5 patches
                patch_distance=10,  # 13x13 search area
                channel_axis=-1)



# fast algorithm
qsm_image = denoise_nl_means(qsm_image, h=0.8 * sigma_est, fast_mode=True,
                                **patch_kw)



# Contrast stretching
p2, p98 = np.percentile(qsm_image, (1, 99.8))
print(p2)
qsm_image = skimage.exposure.rescale_intensity(qsm_image, in_range=(p2, p98), out_range=(-1, 1))
plt.imshow(qsm_image[:,:,100])
plt.show()


# qsm_image = skimage.exposure.equalize_adapthist(qsm_image, clip_limit=0.01, kernel_size=10)
# plt.imshow(qsm_image[:,:,100])
# plt.show()


nlmean_image = nib.Nifti1Image(qsm_image, nib.load(input_image).affine)
nib.save(nlmean_image, "/home/pabaua/dev_tpil/nl_means.nii.gz")
# Step 1: Preprocessing
# Noise reduction using Gaussian blur
#smoothed_image = gaussian(qsm_image, sigma=1)
#plt.imshow(smoothed_image[:,:,100])
#plt.show()

# Contrast enhancement using CLAHE
#contrast_enhanced_image = equalize_adapthist(smoothed_image)
#plt.imshow(contrast_enhanced_image[:,:,100])
#plt.show()

# Step 2: Vessel Enhancement
# Enhancing vessels using Frangi filter
vessel_enhanced_image = frangi(qsm_image, sigmas=(0.5, 1.5, 10), black_ridges=False, alpha=1, beta=1.5, gamma=100)
plt.imshow(vessel_enhanced_image[:,:,100])
plt.show()

# Contrast stretching
p2, p98 = np.percentile(vessel_enhanced_image, (2, 99.9))
vessel_enhanced_image = skimage.exposure.rescale_intensity(vessel_enhanced_image, in_range=(p2, p98), out_range=(0, 1))
plt.imshow(vessel_enhanced_image[:,:,100])
plt.show()

markers = np.zeros(vessel_enhanced_image.shape, dtype=np.uint)
markers[vessel_enhanced_image > 0.5] = 1
plt.imshow(markers[:,:,100])
plt.show()
# labels = skimage.segmentation.random_walker(vessel_enhanced_image, markers, beta=10, mode='cg_j')
# plt.imshow(labels[:,:,100])
# plt.show()



# Step 3: Segmentation
# Simple thresholding (threshold value might need adjustment based on the image)
# threshold_value = 0.25  # This is an example value; adjust based on your images
# vessel_segmented = vessel_enhanced_image > threshold_value
# plt.imshow(vessel_segmented[:,:,100])
# plt.show()

# Post-processing to remove small objects/noise
# This step might include morphological operations like opening
#selem = disk(2)  # Disk element with radius of 2 pixels
#binary_opening(vessel_segmented, selem)
cleaned_vessels = nib.Nifti1Image(vessel_enhanced_image, nib.load(input_image).affine)
nib.save(cleaned_vessels, output_image)
# labels = nib.Nifti1Image(markers, nib.load(input_image).affine)
# nib.save(labels, "/home/pabaua/dev_tpil/labels.nii.gz")

# Displaying the results
# fig, ax = plt.subplots(1, 4, figsize=(20, 5))
# ax[0].imshow(qsm_image[:,:,100], cmap='gray')
# ax[0].set_title('Original QSM Image')
# ax[1].imshow(contrast_enhanced_image[:,:,100], cmap='gray')
# ax[1].set_title('Contrast Enhanced')
# ax[2].imshow(vessel_enhanced_image[:,:,100], cmap='gray')
# ax[2].set_title('Vessel Enhanced')
# ax[3].imshow(cleaned_vessels[:,:,100], cmap='gray')
# ax[3].set_title('Segmented Vessels')
# for a in ax:
#     a.axis('off')
# plt.show()