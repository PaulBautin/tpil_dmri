#!/usr/bin/env bash

module singularity/3.8

my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/scilus_1.3.0.sif' # or .img

bundles=/home/pabaua/scratch/tpil_dev/results/clbp/22-10-26_bundle_seg/results_bundle/
tracto=/home/pabaua/scratch/tpil_dev/results/clbp/22-09-23_tractoflow_bundling/results/
output=/home/pabaua/scratch/tpil_dev/results/clbp/22-11-03_bundle_qc/

run=/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/tpil_dmri/bundle_segmentation/bundle_mosaic/tree_for_bundle_mosaic.sh

command='bash $run -b $bundles -t $tracto -o $output'

singularity exec $my_singularity_img $command






