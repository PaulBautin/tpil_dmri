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
    ln -s $t/$i/Local_Tracking/${i}*local_tracking*.trk $o/$i/${i}__tractogram.trk

    # Ref image
    ln -s $t/$i/DTI_Metrics/${i}*__fa.nii.gz $o/$i/${i}__ref_image.nii.gz
done
echo "Done"




