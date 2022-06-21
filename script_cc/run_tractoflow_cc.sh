#!/bin/bash

# This would run Tractoflow with the following parameters:
#   - Dti_shells 0, 300 and 1000, Fodf_shells 0, 2000, 3000.
#   - profile bundling => WM seeding


#SBATCH --nodes=1              # --> Generally depends on your nb of subjects.
                               # See the comment for the cpus-per-task. One general rule could be
                               # that if you have more subjects than cores/cpus (ex, if you process 38
                               # subjects on 32 cpus-per-task), you could ask for one more node.
#SBATCH --cpus-per-task=32     # --> You can see here the choices. For beluga, you can choose 32, 40 or 64.
                               # https://docs.computecanada.ca/wiki/B%C3%A9luga/en#Node_Characteristics
#SBATCH --mem=0                # --> 0 means you take all the memory of the node. If you think you will need
                               # all the node, you can keep 0.
#SBATCH --time=24:00:00

#SBATCH --mail-user=paul.bautin@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


my_singularity_img='/home/pabaua/scratch/scil_dev/containers/scilus_1.3.0.sif' # or .sif
my_main_nf='/home/pabaua/scratch/scil_dev/tractoflow/main.nf'
my_input='/home/pabaua/scratch/tpil_dev/data/dataset_sansan_bids'


nextflow run $my_main_nf --bids $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --dti_shells "0 300 1000" --fodf_shells "0 2000 3000" -profile bundling

