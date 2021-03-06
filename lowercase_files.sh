#!/bin/bash
#
# Copies all files and directories into new folder "<DATASET_ROOT_FOLDER>_lowercase"
# and converts all filenames to lowercase in new folder.
# This script should be run beside the dataset root folder. If rename command is not found run:
# 'sudo apt install rename'
#
# Usage:
#   ./lowercase_files <DATASET_ROOT_FOLDER>
#
# Example:
#   ./lowercase_files Data_dMRI
#
# Author: Paul Bautin
###################################################

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Global variables
DATASET_ROOT_FOLDER=$1
echo "DATASET ROOT FOLDER: ${DATASET_ROOT_FOLDER}"

# copy and paste dataset in new repository
cd "${DATASET_ROOT_FOLDER}"
cd ../
mkdir -p "${DATASET_ROOT_FOLDER}_lowercase"
cp -a "${DATASET_ROOT_FOLDER}/." "${DATASET_ROOT_FOLDER}_lowercase/"

# find all files in new directory and lowercase all files
find "${DATASET_ROOT_FOLDER}_lowercase" -type f -exec rename 's/(.*)\/([^\/]*)/$1\/\L$2/' {} \;
