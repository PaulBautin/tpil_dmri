#!/bin/bash

bids_dir='/home/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset_v3/'
output_dir='/home/pabaua/dev_tpil/results/results_qsm/results_qsm_nextqsm'

#sudo docker run -it -v /home/pabaua:/home/pabaua vnmd/qsmxt_6.4.0

qsmxt $bids_dir $output_dir --do_t2starmap --do_qsm --do_r2starmap --auto_yes --qsm_algorithm tgv --masking_algorithm bet