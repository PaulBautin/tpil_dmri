Ubuntu 22.04
============

Update and Upgrade Ubuntu
-------------------------

.. code:: sudo

   sudo apt -y upgrade
   sudo apt install ubuntu-restricted-extras
   sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
   sudo apt install git
   sudo apt install tree

Miniconda
---------

Download latest miniconda linux installer from
`website <https://docs.conda.io/en/latest/miniconda.html#linux-installers>`__
and move to /tmp. /tmp is a good directory to download ephemeral items,
like the miniconda bash script, which you won’t need after running it.

You can now verify the data integrity of the installer with
cryptographic hash verification through the SHA-256 checksum:
``sha256sum miniconda.sh``

You can now run the script: ``bash anaconda.sh`` and say yes to
everything

You can now activate the installation by sourcing the ~/.bashrc file:
``source ~/.bashrc``

You can now create a conda environment (to look up available python
versions type: ``conda search python``):
``conda create --name env_scil python=3.7``

You can activate your new environment by typing:
``conda activate env_scil`` and deactivate by typing:
``conda deactivate``

To remove conda environment deactivate and type:
``conda remove --name env_scil --all``

To list all environments type: ``conda info --envs``

To activate conda env in shell script use command:

::

   source /home/pabaua/miniconda3/etc/profile.d/conda.sh
   conda init bash
   conda activate env_scil

Nextflow
--------

Install java by running command: ``sudo apt install default-jre`` and
``sudo apt install default-jdk``

Download latest nextflow version (not edge version) from `Github
releases <https://github.com/nextflow-io/nextflow/releases>`__

Make the binary executable on your system by running
``chmod +x nextflow``.

Optionally, move the nextflow file to a directory accessible by your
$PATH variable (this is only required to avoid remembering and typing
the full path to nextflow each time you need to run it). Example:
``sudo mv ~/Downloads/nextflow /usr/local/bin``

You can temporarily switch to a specific version of Nextflow by
prefixing the nextflow command with the NXF_VER environment variable.
For example: ``NXF_VER=20.04.0 nextflow run``

Singularity
-----------

Singularity is used to package scientific software and deploy that
package to different clusters having the same environment.

Install singularity with the debian package on the `Github repository
releases <https://github.com/sylabs/singularity/releases>`__ then run
download folder: ``sudo dpkg -i singularity-ce_3.10.0-jammy_amd64.deb``

How to build singularity image on CC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter command line: ``singularity remote login``

Then enter access token from: https://cloud.sylabs.io/dashboard

This will allow to build singularity image without sudo using command

singularity build --remote .img .def

Directories
-----------

Create directories in HOME: ``mkdir dev_tpil dev_scil``

Create directories in Documents: ``mkdir tpil``

Create directories in tpil:
``mkdir admin data results reports misc manuscripts biblio conferences``

All result and report files should be written under same format.
Example: ``YYYY-MM-DD_tractoflow_output`` or ``YYYY-MM-DD_qc_rbx``

ANTs
----

To install ANTs, cmake must first be installed ``conda install cmake``

It is then possible to use cmake to configure, build and install ANTs by
running commands:

::

   git clone https://github.com/ANTsX/ANTs.git
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

Fsleyes
-------

Install Fsleyes as part of FSL or install with conda (with an
independant environment) by typing:
``conda install -c conda-forge -n env_fsleyes fsleyes``

To speed up the launching add alias to .bashrc:
``alias fsleyes="conda activate env_fsleyes && fsleyes"``

DCM2BIDS
--------

Installation of dcm2bids can be done through conda (with new conda
environment): ``conda install -c conda-forge -n env_dcm2bids dcm2bids``

Also install dcm2niix with conda (same environmnent):
``conda install -c conda-forge dcm2niix``

Create scaffolding with command: ``dcm2bids_scaffold``

In code create configuration file with command:
``gedit dcm2bids_config.json``

Datalad
-------

Installation of Datalad can be done with:
``sudo apt-get install datalad``

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

Git
---

To execute a git pull from a different directory run command:
``git -C <git_dir> pull``

Globus
------

Globus Connect Personal enables you to share and transfer files to and
from your Linux laptop or desktop computer.

Open a terminal and find: ``cd globusconnectpersonal-x.y.z``

Then run command: ``./globusconnectpersonal``

