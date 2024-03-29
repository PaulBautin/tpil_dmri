source /home/pabaua/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate env_scil

"""
This will result in a binary mask where e,
 and 90% of the
population having at least 20mm of average streamlines length.
"""
# each node with a value of 1 represents a node with at least 90% of the population having at least 1 streamline
scil_filter_connectivity.py /home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_streamline.npy \
    --greater_than /home/pabaua/dev_tpil/data/22-11-16_connectflow/control/*/Compute_Connectivity/sc.npy 1 0.90 -f -v

# 90% of the population is similar to the average (2mm)
scil_filter_connectivity.py /home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_avg.npy \
    --lower_than /home/pabaua/dev_tpil/data/22-11-16_connectflow/control/*/Compute_Connectivity/sim.npy 2 0.90 -f -v

# 90% of the population having at least 20mm of average streamlines length
scil_filter_connectivity.py /home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_len.npy \
    --greater_than /home/pabaua/dev_tpil/data/22-11-16_connectflow/control/*/Compute_Connectivity/len.npy 10 0.90 -v -f
