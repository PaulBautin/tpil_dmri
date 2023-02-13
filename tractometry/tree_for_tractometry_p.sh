#!/usr/bin/env bash
usage() { echo "$(basename $0) [-r RBX] [-t tractoflow/results] [-n noddi/results] [-f freewater_flow/results] [-b new_bundle] [-o output]" 1>&2; exit 1; }

while getopts "r:t:n:f:b:o:" args; do
    case "${args}" in
        r) r=${OPTARG};;
        t) t=${OPTARG};;
        n) n=${OPTARG};;
        f) f=${OPTARG};;
        b) b=${OPTARG};;
        o) o=${OPTARG};;
        *) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${r}" ] || [ -z "${t}" ] || [ -z "${o}" ]; then
    usage
fi

echo "rbx_flow results folder: ${r}"
echo "tractoflow results folder: ${t}"
echo "noddi_flow results folder: ${n}"
echo "freewater_flow results folder: ${f}"
echo "new bundle results folder: ${n}"
echo "Output folder: ${o}"

echo "Building tree for the following folders:"
cd ${t}
for i in *;
do
    echo $i
    mkdir -p $o/$i/bundles
    mkdir -p $o/$i/metrics
    mkdir -p $o/$i/centroids
    ln -s $b/$i/Compute_Centroid/*centroid.trk $o/$i/centroids/

    # if centroids are there, create dir
    #if [ -d "$r/$i/Transform_Centroids/" ]
    #then
	  #  mkdir -p $o/$i/centroids
	  #  ln -s $r/$i/Transform_Centroids/*.trk $o/$i/centroids/
    #fi

    # RBX results
    #for file in $r/$i/Clean_Bundles/*cleaned.trk;
    #do
	  #  name=${file/*Clean_Bundles\//}
	  #  ln -s $file $o/$i/bundles/${name/_m_cleaned.trk/.trk}
    #done

    # tractoflow metrics
    ln -s $t/$i/DTI_Metrics/*fa.nii.gz $o/$i/metrics/fa.nii.gz
    ln -s $t/$i/DTI_Metrics/*ad.nii.gz $o/$i/metrics/ad.nii.gz
    ln -s $t/$i/DTI_Metrics/*md.nii.gz $o/$i/metrics/md.nii.gz
    ln -s $t/$i/DTI_Metrics/*rd.nii.gz $o/$i/metrics/rd.nii.gz
    ln -s $t/$i/FODF_Metrics/*afd_total.nii.gz $o/$i/metrics/afd_total.nii.gz
    ln -s $t/$i/FODF_Metrics/*nufo.nii.gz $o/$i/metrics/nufo.nii.gz

    # noddi_flow metrics
    if [[ ! -z "${n}" ]];
    then
      ln -s $n/$i/Compute_NODDI/*OD.nii.gz $o/$i/metrics/noddi_od.nii.gz
      ln -s $n/$i/Compute_NODDI/*ISOVF.nii.gz $o/$i/metrics/noddi_isovf.nii.gz
      ln -s $n/$i/Compute_NODDI/*ICVF.nii.gz $o/$i/metrics/noddi_icvf.nii.gz
    fi

    # freewater_flow metrics
    if [[ ! -z "${f}" ]];
    then
      ln -s $f/$i/Compute_FreeWater/*FW.nii.gz $o/$i/metrics/freewater.nii.gz
      ln -s $f/$i/FW_Corrected_Metrics/*fa.nii.gz $o/$i/metrics/FAt.nii.gz
      ln -s $f/$i/FW_Corrected_Metrics/*ad.nii.gz $o/$i/metrics/ADt.nii.gz
      ln -s $f/$i/FW_Corrected_Metrics/*rd.nii.gz $o/$i/metrics/RDt.nii.gz
    fi
    
    # if new bundle add
    if [ -d "$b/$i/" ]
    then
	  #  ln -s $b/$i/Compute_Centroid/*centroid.trk $o/$i/centroids/
	    ln -s $b/$i/Filter_tractogram/*cleaned.trk $o/$i/bundles/
    fi
done
echo "Done"

rm -rf ${o}/Average_Bundles
