#!/usr/bin/env bash

usage() { echo "$(basename $0) [-t tractoflow/results] [-b new_bundle/results_bundle] [-o output]" 1>&2; exit 1; }

module load singularity/3.8

my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/scilus_1.3.0.sif' # or .img

while getopts "t:b:o:" args; do
    case "${args}" in
        t) t=${OPTARG};;
        b) b=${OPTARG};;
        o) o=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${t}" ] || [ -z "${b}" ] || [ -z "${o}" ]; then
    usage
fi

echo "tractoflow results folder: ${t}"
echo "Output folder: ${o}"
echo "Building tree for the following folders:"
cd ${b}
for i in *;
do
    echo $i
    mkdir -p $o/results_multi/$i

    # Tractogram
    tractogram27=$b/$i/Filter_tractogram/${i}*cleaned_27.trk
    tractogram45=$b/$i/Filter_tractogram/${i}*cleaned_45.trk
    tractogram47=$b/$i/Filter_tractogram/${i}*cleaned_47.trk
    # Ref image
    ref_image=$t/$i/DTI_Metrics/${i}*__fa.nii.gz
    # Output
    output=$o/results_multi/$i/$i.png

    singularity run -B /home -B /project -B /scratch -B /localscratch:/temp $my_singularity_img scil_visualize_bundles_mosaic.py $ref_image $tractogram27 $tractogram45 $tractogram47 $output
done
echo "Done"






