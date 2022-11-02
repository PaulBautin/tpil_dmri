# -*- coding: utf-8 -*-
import pandas as pd


def load_data_xlsx(file, group_name):
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
    df = df.drop(df.filter(regex='std', axis=1).columns, axis=1)
    #df.columns = df.columns.str.removesuffix("_metric_mean")
    df['group_name'] = group_name
    df['point'] = df['point'].apply(lambda x : int(x))
    return df


def diff_metrics(df_con, df_clbp):
    # prepare dataframe for PCA computation
    df_con_mean = df_con.groupby(["session", "tract", "point"]).mean()
    df_clbp_mean = df_clbp.groupby(["session", "tract", "point"]).mean()
    return df_con_mean.subtract(df_clbp_mean).reset_index()




