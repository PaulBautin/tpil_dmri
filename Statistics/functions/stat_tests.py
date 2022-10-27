# -*- coding: utf-8 -*-
import pandas as pd
import json
import os


def load_data_xlsx_p(file):
    """
    Creates a dataframe based on information present on TractometryFlow output xlsx files
    :param file: xlsx file containing information for statistics. Ex: 'mean_std.xlsx'
    :return sheet_to_df_map: 3D dataframe containing information present in each sheet of xlsx file
    """
    xls = pd.ExcelFile(file)
    sheet_to_df_map = {}
    for sheet_name in xls.sheet_names:
        sheet_to_df_map[sheet_name] = xls.parse(sheet_name)
        sheet_to_df_map[sheet_name].columns = ['ID']+[str(col) for col in sheet_to_df_map[sheet_name].columns[1:]]
    return sheet_to_df_map

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
    print(df)
    #df = pd.concat(df_dict.values())
    #print(df)



def df_gather_metrics(df, group_name, axis_name):
    """
    Load and gather data from the ouput of 'load_data' following the tidy convention.
    :param df: 'load_data' output dataframe. Ex: df_mean_std_con
    :param group_name: identification of dataset group
    :param axis_name: New index naming for tidy format must be a (1x2) array. Ex: [ID, unnamed] -> [subject, tract]
    :return df_all: dataframe containing information present in xlsx file
    """
    df_all = pd.DataFrame()
    for metric in df.keys():
        df[metric] = df[metric].set_index("ID")
        df[metric] = df[metric].stack(dropna=False).rename(metric)
        df_all = pd.concat([df_all, df[metric]], axis=1)
    df_all = df_all.rename_axis(axis_name)
    df_all = df_all.reset_index()
    df_all["group_name"] = group_name
    return df_all


