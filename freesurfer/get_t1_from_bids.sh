#!/usr/bin/env bash
usage() { echo "$(basename $0) [-b bids] [-o output]" 1>&2; exit 1; }

while getopts "b:o:" args; do
    case "${args}" in
        b) b=${OPTARG};;
        o) o=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${b}" ] || [ -z "${o}" ]; then
    usage
fi

echo "bids folder: ${b}"
echo "Output folder: ${o}"

echo "Building tree for the following folders:"
cd ${b}
for i in *;
do
    echo $i
    # t1
    cp ${b}/${i}/ses-v1/anat/*T1w.nii.gz ${o}/${i}_ses-v1_T1w.nii.gz
    cp ${b}/${i}/ses-v2/anat/*T1w.nii.gz ${o}/${i}_ses-v2_T1w.nii.gz
    cp ${b}/${i}/ses-v3/anat/*T1w.nii.gz ${o}/${i}_ses-v3_T1w.nii.gz

done
echo "Done"




