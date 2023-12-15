#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lengthen the streamlines so that they reach deeper in the grey matter.
The lengthened endpoints follow linearly the direction of the last streamline step.
The added length can be specified, and the script requires a grey matter mask
to ensure that the streamline end in the grey matter:
    - if the lengthening gets out of the grey matter, the protruding points will be cut.
    - if despite the lengthening, the endpoint does not reach the grey matter, the
    lengthening will be canceled for that streamline end.
The "with_atlas" option consider that a point is "protruding" (and thus must be shaved)
if the extension changes parcel (or get out of the grey matter). This help avoid the
extension crossing hemishere, for example.
"""


from concurrent.futures import ThreadPoolExecutor
from dipy.io.streamline import load_tractogram, save_tractogram
from nibabel import load
import numpy as np
from dipy.tracking import utils
from scipy.spatial import cKDTree
from time import time


def build_gm_kdtree(grey_matter_data, affine):
    """Build a KD-Tree from the coordinates of the grey matter voxels."""
    grey_matter_coords = utils.apply_affine(affine, np.argwhere(grey_matter_data > 0))
    return cKDTree(grey_matter_coords)


def check_gm_in_range_kdtree(point, kdtree, extension_radius_mm):
    """Use KD-Tree to check if grey matter is within range of a point."""
    closest_dists, _ = kdtree.query(point, distance_upper_bound=extension_radius_mm)
    return np.isfinite(closest_dists).any()


def extend_streamlines_into_grey_matter(tractogram_file, grey_matter_atlas_file, output_file, extension_length_mm=5, with_atlas=True):
    """Extend tractogram streamlines into a given grey matter atlas."""
    
    print("Loading tractogram and grey matter atlas...")
    tractogram = load_tractogram(tractogram_file, grey_matter_atlas_file)
    grey_matter_img = load(grey_matter_atlas_file)
    grey_matter_data = grey_matter_img.get_fdata()

    # Precompute reusable values
    affine_inv = np.linalg.inv(grey_matter_img.affine)
    num_points = int(extension_length_mm * 2)
    step_size = extension_length_mm / num_points

    # Build KD-Tree
    #kdtree = build_gm_kdtree(grey_matter_data, grey_matter_img.affine)

    def extend_single_streamline(streamline):
        """Extend a single streamline."""
        resampled_streamline = streamline.copy()  # Deep copy to ensure independence

        for end_idx, neighbor_idx in [(0, 1), (-1, -2)]:
            # point = resampled_streamline[end_idx]
            # if np.isinf(check_gm_in_range_kdtree(point, kdtree, extension_length_mm)):
            #     break

            direction = (resampled_streamline[neighbor_idx] - resampled_streamline[end_idx]) / np.linalg.norm(resampled_streamline[neighbor_idx] - resampled_streamline[end_idx])

            for i in range(1, num_points + 1):  # Start from 1 to avoid adding the point itself
                new_point = resampled_streamline[end_idx] + i * step_size * direction
                new_point_voxel = utils.apply_affine(affine_inv, new_point).astype(int)

                # Validate new point
                within_image = np.all(new_point_voxel >= 0) and np.all(new_point_voxel < grey_matter_data.shape)
                within_grey_matter = within_image and grey_matter_data[tuple(new_point_voxel)] > 0
                
                within_same_parcel = True  # Default to True for the case when with_atlas is False
                if with_atlas and within_grey_matter:
                    old_point_voxel = utils.apply_affine(affine_inv, resampled_streamline[end_idx]).astype(int)
                    within_same_parcel = (grey_matter_data[tuple(old_point_voxel)] == grey_matter_data[tuple(new_point_voxel)]) or (grey_matter_data[tuple(old_point_voxel)] == 0 and grey_matter_data[tuple(new_point_voxel)] != 0)

                if within_grey_matter and within_same_parcel:
                    resampled_streamline = np.vstack([resampled_streamline, new_point]) if end_idx != 0 else np.vstack([new_point, resampled_streamline])
                    break

        return resampled_streamline

    print("Computing all extensions...")
    with ThreadPoolExecutor() as executor:
        extended_streamlines = list(executor.map(extend_single_streamline, tractogram.streamlines))

    # Update and save the tractogram
    print("Saving streamlines...")
    tractogram.streamlines = extended_streamlines
    save_tractogram(tractogram, output_file)


def main():
    # Example usage
    input_trk="/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl010_ses-v1/Tractography_registration/sub-pl010_ses-v1__pft_tracking_prob_wm_seed_0_t1.trk"
    output_trk="/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl010_ses-v1/Tractography_registration/sub-pl010_ses-v1__pft_tracking_prob_wm_seed_0_t1_ext.trk"
    #input_trk="/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Tractography_filtering/sub-pl007_ses-v1__pft_tracking_prob_wm_seed_0_t1__NAc_proj.trk"
    #output_trk="/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Tractography_filtering/sub-pl007_ses-v1__pft_tracking_prob_wm_seed_0_t1__NAc_proj_ext.trk"
    gm="/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl010_ses-v1/Parcels_to_subject/BN_atlas_first.nii.gz"
    start_time = time() 
    extend_streamlines_into_grey_matter(input_trk, gm, output_trk, extension_length_mm=2)
    stop_time = time()  # Record the stop time
    print(f"Stop time: {stop_time}")
    print(f"Total execution time: {stop_time - start_time} seconds")


if __name__ == "__main__":
    main()