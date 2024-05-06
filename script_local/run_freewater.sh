#!/bin/bash

# This would run Freewater_Flow with the following parameters:
#   - bids: clinical data from TPIL lab (27 CLBP and 25 control subjects), if not in BIDS format use flag --input
#   - with-singularity: container image scilus v1.3.0 (runs: dmriqc_flow, tractoflow, recobundleX, tractometry)
#   - with-report: outputs a processing report when pipeline is finished running
#   - Dti_shells 0 and 300 (usually <700), Fodf_shells 0 and 2000 (usually >1200).
#   - profile: bundling, bundling profile will set the seeding strategy to WM as opposed to interface seeding that is usually used for connectomics


my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.6.0.sif' # or .img
my_main_nf='/home/pabaua/dev_scil/freewater_flow/main.nf'
my_input='/home/pabaua/dev_tpil/data/freewater_test/'


NXF_VER=21.10.6 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html


