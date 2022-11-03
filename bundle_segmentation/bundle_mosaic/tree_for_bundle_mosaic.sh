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
    ln -s $b/$i/Filter_tractogram/${i}*27.trk $o/$i/${i}__tractogram27.trk
    ln -s $b/$i/Filter_tractogram/${i}*45.trk $o/$i/${i}__tractogram45.trk
    ln -s $b/$i/Filter_tractogram/${i}*47.trk $o/$i/${i}__tractogram47.trk

    # Ref image
    ln -s $t/$i/DTI_Metrics/${i}*__fa.nii.gz $o/$i/${i}__ref_image.nii.gz
    scil_visualize_bundles_mosaic.py $o/$i/${i}__ref_image.nii.gz $o/$i/${i}__tractogram27.trk $o/results/$i/$i.png -f
done
echo "Done"

for d in $o/*/ ; do
    echo "$d"

done





