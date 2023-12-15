fMRI prep pipeline
================================================================

Outputs
----------------------------------------------------------------

Scilpy is the main library supporting research and development at the
Sherbrooke Connectivity Imaging Lab.

``git clone https://github.com/scilus/scilpy.git``

``cd scilpy``

``pip install -e .``

(Installation of scipy requires ``sudo apt-get install gfortran``)

Tractfoflow
----------------------------------------------------------------

``git clone https://github.com/scilus/tractoflow.git`` in directory scil

``git clone https://github.com/scilus/containers-tractoflow.git`` in
directory scil

``singularity build tractoflow_2_3_0_container_2_2_1.img singularity_tractoflow.def``
in new directory containers

MI-Brain
----------------------------------------------------------------

Download MI-Brain tarball from `Github repository
releases <https://github.com/imeka/mi-brain/releases/tag/2020.04.09>`__
then extract with command:
``tar –xvzf MI-Brain-2020.04.09_r2e0ff5-linux-x86_64.tar.gz``

It is now possible to place MI-Brain in HOME directory with command:
``mv MI-Brain-2020.04.09_r2e0ff5-linux-x86_64 ~/MI-Brain``

To accelerate opening add mibrain alias to .bashrc:
``alias mibrain="bash ~/MI_Brain/MI-Brain.sh"``

dMRI QC
--------------------

Quality control can be run with dmriqc_flow `github
repository <https://github.com/scilus/dmriqc_flow>`__

Clone dmriqc_flow github repository with command:
``git clone https://github.com/scilus/dmriqc_flow.git`` in scil_dev

To build singularity image run command:
``sudo singularity build scilus_1.3.0.sif docker://scilus/scilus:1.3.0``
in scil_dev/containers

Combine flows
----------------------------------------------------------------

First clone the combine_flows `Github
repository <https://github.com/scilus/combine_flows>`__

When preparing the dataset for reconbundleX use command.

::

   bash tree_for_rbx_flow.sh \
   -t <tractoflow/results> \
   -o <output_dir>

We have also created a new script for isolating NAC_mPFC bundle. To
create the tree run:

::

   bash /home/pabaua/scratch/scil_dev/combine_flows/tree_for_new_bundle.sh \
   -r /home/pabaua/scratch/tpil_dev/results/sub-DCL/22-07-11_rbx/results_rbx/ \
   -t /home/pabaua/scratch/tpil_dev/results/sub-DCL/22-07-07_tractoflow/results/ \
   -a /home/pabaua/scratch/scil_dev/atlas/BN_Atlas_246_1mm.nii.gz \
   -o /home/pabaua/scratch/tpil_dev/data/data_new_bundle_DCL

To run tractometry with the output of the new script for isolating
NAC_mPFC bundle, run command:

::

   bash /home/pabaua/scratch/tpil_dev/github/tpil_dmri/tree_for_tractometry_p.sh \
   -r /home/pabaua/scratch/tpil_dev/results/sub-CON/22-07-12_rbx/results_rbx \
   -t /home/pabaua/scratch/tpil_dev/results/sub-CON/22-07-10_tractoflow/results \
   -b /home/pabaua/scratch/tpil_dev/results/sub-CON/22-08-16_NAC_mPFC/results_bundle \
   -o /home/pabaua/scratch/tpil_dev/data/data_tractometry_with_NACmPFC_CON
