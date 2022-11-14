#!/usr/bin/env bash

usage() { echo "$(basename $0) [-t tractoflow/results] [-o output]" 1>&2; exit 1; }

while getopts "t:o:" args; do
    case "${args}" in
        t) t=${OPTARG};;
        o) o=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${t}" ] || [ -z "${o}" ]; then
    usage
fi

echo "tractoflow results folder: ${t}"
echo "Output folder: ${o}"
echo "Building tree for the following folders:"
cd ${t}
for i in *;
do
    echo $i
    mkdir -p $o/$i/

    # Tractogram
    ln -s $t/$i/PFT_Tracking/${i}*pft_tracking*.trk $o/$i/${i}__tractogram.trk
    # Ref image
    ln -s $t/$i/DTI_Metrics/${i}*__fa.nii.gz $o/$i/${i}__ref_image.nii.gz
    ln -s ${t}/${i}/Resample_DWI/${i}__dwi_resampled.nii.gz $o/$i/${i}__dwi.nii.gz
    ln -s ${t}/${i}/Eddy_Topup/${i}__dwi_eddy_corrected.bvec $o/$i/${i}__dwi.bvec
    ln -s ${t}/${i}/Eddy_Topup/${i}__bval_eddy $o/$i/${i}__dwi.bval
    ln -s ${t}/${i}/PFT_Tracking/${i}__pft_tracking_prob_wm_seed_0.trk $o/$i/${i}__pft_tracking_prob_wm_seed_0.trk
    ln -s ${t}/${i}/FODF_Metrics/${i}__fodf.nii.gz $o/$i/${i}__fodf.nii.gz
    ln -s ${t}/${i}/FODF_Metrics/${i}__peaks.nii.gz $o/$i/${i}__peaks.nii.gz
    #ln -s ${t}/${i}/DTI_Metrics/${i}__fa.nii.gz $o/$i/metrics/${i}__fa.nii.gz

done
echo "Done"




