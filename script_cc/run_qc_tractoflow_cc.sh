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
#SBATCH --time=1:00:00

#SBATCH --mail-user=paul.bautin@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load StdEnv/2020 java/14.0.2 nextflow/21.10.3 singularity/3.8


my_singularity_img='/home/pabaua/scratch/scil_dev/containers/scilus_1.3.0.sif' # or .img
my_main_nf='/home/pabaua/scratch/scil_dev/dmriqc_flow/main.nf'
my_input='/home/pabaua/scratch/tpil_dev/results/sub-DCL/22-07-07_tractoflow/results'


NXF_VER=nextflow/21.10.3 nextflow run $my_main_nf -profile tractoflow_qc_all --input $my_input \
    -with-singularity $my_singularity_img -resume
