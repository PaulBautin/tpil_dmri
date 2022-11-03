#!/usr/bin/env bash

usage() { echo "$(basename $0) [-t tractoflow/results] [-b new_bundle/results_bundle] [-o output]" 1>&2; exit 1; }

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
    mkdir -p $o/$i/
    mkdir $o/results/$i

    # Tractogram
    tractogram=$b/$i/Filter_tractogram/${i}*cleaned.trk
    # Ref image
    ref_image=$t/$i/DTI_Metrics/${i}*__fa.nii.gz
    # Output
    output=$o/results/$i/$i.png

    scil_visualize_bundles_mosaic.py $ref_image $ref_image $output
done
echo "Done"






