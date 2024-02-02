#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Connectome Visualization and Smoothing Pipeline

This script provides a pipeline for the following tasks:
1. Mapping high-resolution structural connectivities using the Connectome Spatial Smoothing (CSS) package.
2. Applying a smoothing kernel to the connectome.
3. Filtering the connectome based on a specific brain structure.
4. Saving the modified data into a new CIFTI file.
5. Visualizing the CIFTI data on a surface plot.

Usage:
Run this script in a terminal, supplying the required command-line arguments.
>>> python CCS_2.py <in_tractogram> <surf_lh> <surf_rh> <cifti_file> <out_file>
"""


import pandas as pd
import nibabel as nib
from pathlib import Path
from surfplot import Plot, utils
from neuromaps.datasets import fetch_fslr
import matplotlib.pyplot as plt

def plot_surface(stat_map_lh, stat_map_rh):
    """
    Plot a surface representation of a CIFTI file.

    Parameters
    ----------
    cifti_file : str or nib.Cifti2Image
        The CIFTI file or image object to plot.
    """
    surfaces = fetch_fslr()
    plt.style.use('dark_background')
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=['lateral','medial', 'ventral'], zoom=1.2)
    #stat_map_lh = utils.threshold(stat_map_lh, 0.15)
    #stat_map_rh = utils.threshold(stat_map_rh, 0.15)
    p.add_layer({'left':stat_map_lh, 'right':stat_map_rh}, cbar=True, cmap='jet')
    fig = p.build()
    plt.show(block=True)

# Define the base path to the files
base_path = Path('/home/pabaua/dev_tpil/results/results_article/results')

# List to store data for each file
data_list = []

# Iterate over each .dscalar.nii file in the subdirectories
for dscalar_file in base_path.rglob('sub-*/ses-v*/Save_cifti/sub-*_ses-v*_nac.dscalar.nii'):
    # Extract data from the file
    data = nib.load(dscalar_file).get_fdata()  
    lh_data = [data[0, bm.index_offset:bm.index_offset + bm.index_count] for bm in nib.load(dscalar_file).header.get_index_map(1).brain_models if bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT']
    rh_data = [data[0, bm.index_offset:bm.index_offset + bm.index_count] for bm in nib.load(dscalar_file).header.get_index_map(1).brain_models if bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT']
    # Extract metadata from the file path (e.g., subject ID and session)
    subject_id = [part for part in dscalar_file.parts if part.startswith('sub-')][0]
    session = [part for part in dscalar_file.parts if part.startswith('ses-')][0] 
    # Append the data and metadata to the data list
    data_list.append({
        'subject_id': subject_id,
        'session': session,
        'lh_data': lh_data,
        'rh_data': rh_data})

# Create a DataFrame from the data list
df = pd.DataFrame(data_list)

# Now `df` contains your compiled data, with one row per file and columns for subject_id, session, and data
print(df)

# Compile dscalars
#df = df[df['subject_id'] == 'sub-010']
#print(df)
plot_surface(df['lh_data'].sum()[0], df['rh_data'].sum()[0])
print(df.sum())




# def main():
#     """
#     Main function to execute the pipeline for connectivity mapping and visualization.
#     """
#     # Define the base path to the files
#     base_path = '/home/pabaua/dev_tpil/results/results_article/results'
#     base_path.glob('sub-*/ses-v*/Save_cifti/sub-*_ses-v*_nac.dscalar.nii')
    
# if __name__ == "__main__":
#     main()
