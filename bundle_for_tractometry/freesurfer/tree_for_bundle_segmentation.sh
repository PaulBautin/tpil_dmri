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

if [ -z "${t}" ] || [ -z "${a}" ] || [ -z "${o}" ]; then
    usage
fi

echo "tractoflow results folder: ${t}"
echo "freesurfer_flow output folder: ${a}"
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

    # atlas
    ln -s $a/$i/Generate_Atlases_FS_BN_GL_SF/atlas_brainnetome_v4.nii.gz $o/$i/${i}__labels.nii.gz

    # tranformations
    ln -s $t/$i/Register_T1/${i}*__output1Warp.nii.gz $o/$i/${i}__output1Warp.nii.gz
    ln -s $t/$i/Register_T1/${i}*__output0GenericAffine.mat $o/$i/${i}__output0GenericAffine.mat
done
echo "Done"




