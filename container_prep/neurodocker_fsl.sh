#!/bin/bash

set -e

neurodocker generate singularity \
	--pkg-manager apt \
   	--base-image debian:bullseye-slim \
	--fsl version=6.0.5.1 \
   	 > /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_fsl


sudo singularity build /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_fsl.simg /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_fsl
