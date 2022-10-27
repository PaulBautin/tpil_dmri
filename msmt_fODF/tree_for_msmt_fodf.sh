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

echo "tractoflow folder: ${t}"
echo "Output folder: ${o}"

echo "Building tree for the following folders:"
cd $t
for i in *[!{FRF}];
do
   echo $i
   mkdir -p $o/$i

   ln -s ${t}/${i}/Resample_DWI/*dwi_resampled.nii.gz ${o}/${i}/dwi.nii.gz
   ln -s ${t}/${i}/Extract_B0/*b0_mask_resampled.nii.gz ${o}/${i}/brain_mask.nii.gz
   ln -s ${t}/${i}/Eddy*/*bval_eddy ${o}/${i}/bval
   ln -s ${t}/${i}/Eddy*/*dwi_eddy_corrected.bvec ${o}/${i}/bvec
done

rm -rf ${o}/Readme*
rm -rf ${o}/Read_BIDS
