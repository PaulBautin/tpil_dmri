#!/usr/bin/env bash

usage() { echo "$(basename $0) [-t tractoflow/results] [-a atlas] [-o output]" 1>&2; exit 1; }

while getopts "t:a:o:" args; do
    case "${args}" in
        t) t=${OPTARG};;
        a) a=${OPTARG};;
        o) o=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${a}" ] || [ -z "${t}" ] || [ -z "${o}" ]; then
    usage
fi

echo "tractoflow results folder: ${t}"
echo "atlas folder: ${a}"
echo "Output folder: ${o}"

# Atlas
ln -s $a $o/

echo "Building tree for the following folders:"
cd ${t}
for i in *;
do
    echo $i
    mkdir -p $o/$i/ref_image
    mkdir -p $o/$i/tractogram

    # Tractogram
    ln -s $t/$i/Local_Tracking/${i}*local_tracking*.trk $o/$i/tractogram/${i}__tractogram.trk

    # Ref image
    ln -s $t/$i/DTI_Metrics/${i}*__fa.nii.gz $o/$i/ref_image/${i}__ref_image.nii.gz
done
echo "Done"




