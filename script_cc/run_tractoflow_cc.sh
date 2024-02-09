#!/bin/bash

# This would run Tractoflow with the following parameters:
#   - bids: clinical data from TPIL lab (27 CLBP and 25 control subjects), if not in BIDS format use flag --input
#   - with-singularity: container image scilus v1.4.0 (runs: dmriqc_flow, tractoflow, recobundleX, tractometry)
#   - with-report: outputs a processing report when pipeline is finished running
#   - Dti_shells 0 and 1000 (usually <1200), Fodf_shells 0 1000 and 2000 (usually >700, multishell CSD-ms).
#   - profile: bundling, bundling profile will set the seeding strategy to WM as opposed to interface seeding that is usually used for connectomics

#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=48:00:00

#SBATCH --mail-user=paul.bautin@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL



module load StdEnv/2020 java/14.0.2 nextflow/22.10.8 apptainer/1.1.8

git -C /home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/tractoflow checkout tags/2.4.3

my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/scilus_1.6.0.sif' # or .img
my_main_nf='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/tractoflow/main.nf'
my_input='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset/'
my_bidsignore='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset/.bidsignore'


export NXF_CLUSTER_SEED=$(shuf -i 0-16777216 -n 1)

nextflow run $my_main_nf --bids $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --dti_shells "0 1000" --fodf_shells "0 1000 2000" -profile bundling --run_gibbs_correction true \
    --bidsignore $my_bidsignore  -profile use_gpu

