#!/bin/bash

set -e

neurodocker generate singularity \
	--pkg-manager apt \
   	--base-image debian:bullseye-slim \
	--ants version=2.3.4 \
   	 > /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_trimeshpy


sudo singularity build /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_trimeshpy.simg /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_trimeshpy
