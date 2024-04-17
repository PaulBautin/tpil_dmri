#!/bin/bash

set -e

neurodocker generate singularity \
	--pkg-manager apt \
   	--base-image debian:bullseye-slim \
    --install git pip \
	--run-bash 'git clone https://github.com/CHrlS98/aodf-toolkit.git && mv aodf-toolkit /opt/ && pip install --upgrade pip && pip install -e /opt/aodf-toolkit/.' \
   	 > /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_aodf_toolkit


sudo singularity build /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_aodf_toolkit.simg /home/pabaua/dev_tpil/tpil_dmri/container_prep/singularity_aodf_toolkit
