
from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Present statistics on control vs. chronic pain patients
#
# example: python tractometry_stat.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import json
from glob import glob

import pandas as pd
import numpy as np
import os
import argparse
import json
import scipy.io


# Parser
#########################################################################################

def get_parser():
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Compute statistics based on the .xlsx files containing the tractometry metrics:",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-func",
        required=True,
        default='func_results',
        help='Path to folder that contains output .mat files (e.g. "22-07-13_tractometry_CLBP/Statistics")',
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-fig',
        help='Generate figures',
        action='store_true'
    )
    optional.add_argument(
        '-o',
        help='Path where figures will be saved. By default, they will be saved in the current directory.',
        default="."
    )
    return parser


def main():
    """
    main function, gather stats and call plots
    """
    pd.options.display.width = 0

    ### Get parser elements
    parser = get_parser()
    arguments = parser.parse_args()
    path_func = os.path.abspath(os.path.expanduser(arguments.func))
    path_output = os.path.abspath(arguments.o)

    # Get all data files and read matlab file
    func_mat = scipy.io.loadmat(path_func)
    print(func_mat.items())
    print(func_mat['V2V3'].shape)
    df_func = pd.DataFrame.from_dict(func_mat)

if __name__ == "__main__":
    main()