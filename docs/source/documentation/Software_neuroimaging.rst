NeuroImaging Software
=====================

.. _ants-1:

ANTs
----

To install ANTs, cmake must first be installed ``conda install cmake``

It is then possible to use cmake to configure, build and install ANTs by
running commands:

.. code:: git

   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX=/opt/ANTs ../ANTs
   make
   cd ANTS-build
   sudo make install

Assuming your install prefix was /opt/ANTs, there will now be a binary
directory /opt/ANTs/bin, containing the ANTs executables and scripts.

::

   export ANTSPATH=/opt/ANTs/bin/
   export PATH=${ANTSPATH}:$PATH

.. _fsleyes-1:

Fsleyes
-------

Install Fsleyes as part of FSL or install with conda (with an
independant environment) by typing:
``conda install -c conda-forge -n env_fsleyes fsleyes``

To speed up the launching add alias to .bashrc:
``alias fsleyes="conda activate env_fsleyes && fsleyes"``

.. _dcm2bids-1:

DCM2BIDS
--------

Installation of dcm2bids can be done through conda (with new conda
environment): ``conda install -c conda-forge -n env_dcm2bids dcm2bids``

Also install dcm2niix with conda (same environmnent):
``conda install -c conda-forge dcm2niix``

Create scaffolding with command: ``dcm2bids_scaffold``

In code create configuration file with command:
``gedit dcm2bids_config.json``

.. _template-flow-1:

Template flow
-------------

The TemplateFlow Archive aggregates all the templates for
redistribution. The archive uses `DataLad <https://datalad.org/>`__ to
maintain all templates under version control.

-  To install template flow with Datalad run command:
   ``datalad install -r ///templateflow``. Usage example:

::

   cd templateflow
   datalad get -r tpl-MNI152NLin2009cAsym

-  To install template flow with Python run command:
   ``python3 -m pip install templateflow`` Usage example:

::

   from templateflow import api as tflow
   tflow.get('MNI152NLin6Asym', desc=None, resolution=1, suffix='T1w', extension='nii.gz')

Neurodocker
-----------

Neurodocker is a command-line program that generates custom Dockerfiles
and Singularity recipes for neuroimaging. Neurodocker can be installed
in a conda environment (using pip).

::

   conda create -n neurodocker python pyyaml
   conda activate neurodocker
   python -m pip install neurodocker
   neurodocker --help

To use neurodocker with singularity first create a .def file with
command (example: Ants, Freesurfer, FSL)

::

   neurodocker generate singularity \
   --pkg-manager apt \
   --base-image debian:stretch \
   --ants version=2.3.2 \
   --freesurfer version=7.3.1 \
   --fsl version=6.0.5.1 \
   > singularity_container

It is then possible to create a singularity image:
``sudo singularity build sing_container.sif singularity_container``

La construction du container peut prendre plusieurs heures --
particulierement FSL step 2 (a roulé pendant 1 nuit entière). Maybe
change ``export ANTSPATH="/opt/ants-2.3.2/"`` to
``export ANTSPATH="/opt/ants-2.3.2/bin/"``

niprep
------

smriprep
~~~~~~~~

Singularity installation
``singularity build smriprep_img.simg docker://nipreps/smriprep``

Freesurfer
----------

Download latest release from
`site <https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads>`__. It
is recommend using the \*.rpm packages to install freesurfer on
CentOS/RedHat linux, the \*.deb package for Ubuntu and and the \*.pkg
installer for MacOS. To speed up, add to bashrc the FREESURFER_HOME env
variable that freesurfer relies on:

::

   alias setupfs='source $FREESURFER_HOME/SetUpFreeSurfer.sh'

Do not forget to export SUBJECTS_DIR before running freesurfer.

::

   export SUBJECTS_DIR=/home/dev_tpil/freesurfer_results
