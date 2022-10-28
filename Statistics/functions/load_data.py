# -*- coding: utf-8 -*-
import pandas as pd


def load_data_xlsx(file):
    """
    Creates a dataframe based on information present on TractometryFlow output xlsx files
    :param file: xlsx file containing information for statistics. Ex: 'mean_std.xlsx'
    :return sheet_to_df_map: 3D dataframe containing information present in each sheet of xlsx file
    """
    # reads all sheets and stores dataframes in a dictionary
    df_dict = pd.read_excel(file, sheet_name=None, index_col=0)
    df = pd.concat(df_dict, axis=1).stack().rename_axis(['subject', 'tract']).reset_index()
    df[['tract', 'point']] = df['tract'].str.rsplit('_', 1, expand=True)
    df[['subject', 'session']] = df['subject'].str.rsplit('_ses-', 1, expand=True)
    return df
