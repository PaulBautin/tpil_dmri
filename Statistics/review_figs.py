
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


import os
import pandas as pd
import plotly.express as px
import numpy as np
import pdfkit
from xhtml2pdf import pisa             # import python module

def main():
    # Define the absolute path to the Excel file
    path_results = os.path.abspath("/home/pabaua/Downloads/Table1_review_final_2.xlsx")
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(path_results, sheet_name="Sheet14", header=1, usecols=range(7))
    print(df)
    
    # Create a treemap using Plotly Express
    fig = px.treemap(
        df,
        path=[
            px.Constant("dMRI CP review"),
            "IASP classification",
            'Type of chronic pain',
            'Analysis method',
            "Authors & year",
            "dMRI sequence details (# of directions, b value (s/mm2, resolution))"
        ],
        hover_name="Authors & year",
        custom_data=[
            "dMRI sequence details (# of directions, b value (s/mm2, resolution))",
            "Analysis method",
            "Main findings"
        ]
    )
    
    # Update the root color of the treemap
    fig.update_traces(root_color="lightgrey")
    
    # Update the layout of the treemap
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    
    # Update the text template to display custom data
    fig.update_traces(texttemplate=(
        "<b>dMRI sequence:</b> %{customdata[0]}<br>"
        "<b>Analysis method:</b> %{customdata[1]}<br>"
        "<b>Main findings:</b> %{customdata[2]}"
    ))
    # Display the treemap
    fig.show()
    #fig.write_image("/home/pabaua/Downloads/review_fig_2.html")

    

    # Utility function
    def convert_html_to_pdf(source_html, output_filename):
        # open output file for writing (truncated binary)
        result_file = open(output_filename, "w+b")

        # convert HTML to PDF
        pisa_status = pisa.CreatePDF(
                source_html,                # the HTML to convert
                dest=result_file)           # file handle to recieve result

        # close output file
        result_file.close()                 # close output file

        # return True on success and False on errors
        return pisa_status.err
    convert_html_to_pdf('/home/pabaua/Downloads/dMRI_chronic_pain_review_figure_2.html', '/home/pabaua/Downloads/review_fig_2.pdf')

if __name__ == "__main__":
    main()
