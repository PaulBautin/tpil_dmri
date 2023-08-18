First you must connect to compute canada with command
``ssh userID@graham.computecanada.ca``. To not type full command each
time you can define a bash alias in $HOME/.bashrc :
``alias graham="ssh userID@graham.computecanada.ca"``

Loading a Python module
-----------------------

the first step is to make sure you have the right python version
installed

``module avail python``

for example you can choose (python>3.6) ex: ``module load python/3.7.4``

Loading Singularity
-------------------

To enable the use of Singularity on CC systems:
``module load singularity``

Virtual environment
-------------------

With each version of python CC provides the virtualenv tool that allows
users to create virtual environments within which you can easily install
Python packages.

To create a virtual environment, make sure you have selected a Python
version with module load python as shown above in section Loading a
Python module. If you expect to use any of the packages listed in
section SciPy stack above, also run ``module load scipy-stack``. Then
enter the following command, where ENV is the name of the directory for
your new environment: ``virtualenv --no-download ENV``

Once the virtual environment has been created, it must be activated:
``source ENV/bin/activate``

You should also upgrade pip in the environment:
``pip install --no-index --upgrade pip``

To exit the virtual environment, simply enter the command deactivate:
``deactivate``

File transfer
-------------

Globus is the preferred tool for transferring data between Compute
Canada systems, and if it can be used, it should.

Run command
-----------

To launch the sbatch: ``sbatch -A <def-user> <script_name>.sh``

To check if it has been launched: ``squeue -u <USER>``

To quit the cluster: ``exit``

Disk usage
----------

To analyse disk usage, use diskusage_report to see if you are at or over
your quota: ``diskusage_report``

Interactive job
---------------

To request interactive processors, use:
``salloc --time=DD-HH:MM --mem-per-cpu=<number>G --ntasks=<number> --account=<your_account>``

Set the OMP_NUM_THREADS appropriately
-------------------------------------

::

   if [ -z "${SLURM_CPUS_PER_TASK+x}" ]; then
       export OMP_NUM_THREADS=1
   else
       export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
   fi
