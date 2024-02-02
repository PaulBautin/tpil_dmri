import json
from pathlib import Path

base_path = Path('/home/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset_copy')

# Iterate over each .dscalar.nii file in the subdirectories
#for json_file in base_path.rglob('sub-*/ses-v*/fmap/sub-*_ses-v*_acq-rest_dir-AP_epi.json'):
for json_file in base_path.rglob('sub-*/ses-v*/func/sub-*_ses-v*_task-rest_bold.json'):

    data = {
        # Other keys and values would go here.
    }

    slice_timing = [0,
                0.5375,
                0.0895833333333333,
                0.627083333333333,
                0.179166666666667,
                0.716666666666667,
                0.985416666666667,
                0.447916666666667,
                0.895833333333333,
                0.358333333333333,
                0.80625,
                0.26875,
                0,
                0.5375,
                0.0895833333333333,
                0.627083333333333,
                0.179166666666667,
                0.716666666666667,
                0.985416666666667,
                0.447916666666667,
                0.895833333333333,
                0.358333333333333,
                0.80625,
                0.26875,
                0,
                0.5375,
                0.0895833333333333,
                0.627083333333333,
                0.179166666666667,
                0.716666666666667,
                0.985416666666667,
                0.447916666666667,
                0.895833333333333,
                0.358333333333333,
                0.80625,
                0.26875,
                0,
                0.5375,
                0.0895833333333333,
                0.627083333333333,
                0.179166666666667,
                0.716666666666667,
                0.985416666666667,
                0.447916666666667,
                0.895833333333333,
                0.358333333333333,
                0.80625,
                0.26875]


    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # Initialize 'data' as an empty dictionary if the file does not exist.
        data = {}

    # Check if 'SliceTiming' exists in the data, and append the new values if it does.
    # If 'SliceTiming' does not exist, create it with the new values.
    if 'SliceTiming' in data:
        data['SliceTiming'] = slice_timing  # Append new values to existing 'SliceTiming'.
    else:
        data['SliceTiming'] = slice_timing  # Create 'SliceTiming' with new values.

        

    # Serialize 'data' to a JSON formatted string and write it to a file.
    with open(json_file, 'w') as file:
        #data = json.load(json_file)
        # Add 'SliceTiming' to the 'data' dictionary.
        json.dump(data, file, indent=4)