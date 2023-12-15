
from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Present figures for dMRI review
#
# example: python tractometry_stat.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Marc-Antoine Fortier
#
# About the license: see the file LICENSE
#########################################################################################


import json
import glob

import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

def fig_c_sunburst():
    # Sunbrust table
    path_results = os.path.abspath("/home/pabaua/Downloads/Table1_review_final_2 (1).xlsx")
    df = pd.read_excel(path_results, sheet_name="Fig c", header=0, usecols="I:L",nrows=30)
    print(df)
    
    # Sample data in a DataFrame
    data = {
        'ids': ["NA", "Elec", "HG", "Elec_Phone", "Elec_Laptop", "HG_Furniture", "HG_Phone"],
        'labels': ["North America", "Electronics", "Home Goods", "Phones", "Laptops", "Furniture", "Phones"],
        'parents': ["", "NA", "NA", "Elec", "Elec", "HG", "HG"],
        'counts': [100, 60, 40, 30, 30, 20, 20]
    }
    #df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Sunburst(
        ids=df.ids,
        labels=df.labels,
        parents=df.parents,
    ))

    fig.show()
    

fig_c_sunburst()



# Helper function to create hierarchical ids
def create_id(class_name, tract_region, reported_finding, detailed_region):
    components_class_tracts = [str(x) for x in [class_name, tract_region] if not pd.isna(x)]
    components_tract_findings = [str(x) for x in [tract_region, reported_finding] if not pd.isna(x)] 
    components_findings_detailed = [str(x) for x in [reported_finding, detailed_region] if not pd.isna(x)]
    
    return '-'.join(components_class_tracts), '-'.join(components_tract_findings), '-'.join(components_findings_detailed)

def create_table():
    # Original DataFrame
    path_results = os.path.abspath("/mnt/c/Users/mafor/Downloads/Table1_review_final_2.xlsx")
    df = pd.read_excel(path_results, sheet_name="Fig c", header=0, usecols="A:E")
    # Initialize the hierarchy data for the sunburst plot
    class_tracts_data = []
    tract_findings_data = []
    findings_detailed_data = []
    class_data = []
    tract_data = []
    findings_data = []
    detailed_data = []

    # Loop through your original DataFrame to create the Sunburst data
    for index, row in df.iterrows():
        class_name = row['Class']
        tract_region = row['Tracts and regions']
        reported_finding = row['Reported findings']
        detailed_region = row['Detailed regions']
        class_data.append(class_name)
        tract_data.append(tract_region)
        findings_data.append(reported_finding)
        detailed_data.append(detailed_region)

        class_tracts, tract_findings, findings_detailed = create_id(class_name, tract_region, reported_finding, detailed_region)
        class_tracts_data.append(class_tracts)
        tract_findings_data.append(tract_findings)
        findings_detailed_data.append(findings_detailed)
    
    class_tracts_df = pd.DataFrame(zip(class_tracts_data, class_data, tract_data), columns=['class_tracts','class','tract'])
    tract_findings_df = pd.DataFrame(zip(tract_findings_data, tract_data, findings_data), columns=['tract_findings','tract', 'findings'])
    findings_detailed_df = pd.DataFrame(zip(findings_detailed_data, findings_data, detailed_data), columns=['findings_detailed', 'findings', 'detailed'])

    class_tracts_df.to_csv('/home/mafor/dev_tpil/review_dMRI/class_tracts_df.csv')
    tract_findings_df.to_csv('/home/mafor/dev_tpil/review_dMRI/tracts_findings_df.csv')
    findings_detailed_df.to_csv('/home/mafor/dev_tpil/review_dMRI/findings_detailed_df.csv')