
from __future__ import division

from dipy.align.streamlinear import groupwise_slr
from dipy.viz.streamline import show_bundles, window, actor
from dipy.io.streamline import load_trk, load_tractogram, save_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.tracking.fbcmeasures import FBCMeasures
# Compute lookup table
from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.io.stateful_tractogram import Space, StatefulTractogram

import glob
import os
import logging
import vtk
import pandas as pd
import numpy as np
logging.basicConfig(level=logging.INFO)
from dipy.data import read_five_af_bundles



def main():
    """
    main function, gather stats and call plots
    """
    file_paths = glob.glob(os.path.join('/home/pabaua/dev_tpil/results/results_nac/results/', '*/Tractography_filtering/*__pft_tracking_prob_wm_seed_0__NAc_proj.trk'))
    file_paths_ref = glob.glob(os.path.join('/home/pabaua/dev_tpil/results/results_nac/results/*/Subcortex_registration/*__t1_warped.nii.gz'))
    df_trk = pd.DataFrame.from_dict({os.path.basename(os.path.dirname(os.path.dirname(fp))): fp for fp in file_paths}, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'trk'})
    df_ref = pd.DataFrame.from_dict({os.path.basename(os.path.dirname(os.path.dirname(fp))): fp for fp in file_paths_ref}, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'ref'})
    df = pd.merge(df_trk, df_ref, on='subject')
    df['load_trk'] = df.apply(lambda row: load_trk(row['trk'],row['ref']).streamlines, axis=1)
    bundles = df['load_trk'].values
    colors = [[0.91, 0.26, 0.35], [0.69, 0.85, 0.64], [0.51, 0.51, 0.63]]

    #df['bundles_reg'], aff, d = groupwise_slr(bundles, verbose=True)

    show_bundles(df['load_trk'].values, interactive=False, colors=colors,
                save_as='before_group_registration.png')
    
    print(bundles[0])
    print(df[df['subject'] == 'sub-pl007_ses-v1']['load_trk'].values)
    cc_sft = StatefulTractogram(df[df['subject'] == 'sub-pl007_ses-v1']['load_trk'].values[0], df[df['subject'] == 'sub-pl007_ses-v1']['ref'].values[0], Space.RASMM)
    save_tractogram(cc_sft, 'cc.trk')

    D33 = 1.0
    D44 = 0.02
    t = 1
    k = EnhancementKernel(D33, D44, t)
    fbc = FBCMeasures(df[df['subject'] == 'sub-pl007_ses-v1']['load_trk'].values[0],k)
    # Calculate LFBC for original fibers
    fbc_sl_orig, clrs_orig, rfbc_orig = fbc.get_points_rfbc_thresholded(0, emphasis=0.01)
    print(clrs_orig)
    cc_sft = StatefulTractogram(fbc_sl_orig, df[df['subject'] == 'sub-pl007_ses-v1']['ref'].values[0], Space.RASMM)
    save_tractogram(cc_sft, 'cc.trk')
    print(fbc_sl_orig)

    # show_bundles(fbc_sl_orig, interactive=False,
    #             save_as='before_group_registration.png')
    print(fbc_sl_orig)
    print(fbc)
    # Create scene
    scene = window.Scene()

    # Original lines colored by LFBC
    lineactor = actor.line(fbc_sl_orig, np.vstack(clrs_orig), linewidth=0.2)
    scene.add(lineactor)
    # Show original fibers
    scene.set_camera(view_up=(1, 0, 1))
    window.record(scene, n_frames=1, out_path='OR_before.png', size=(900, 900))
    

if __name__ == "__main__":
    main()