# Load the data from the tsv files from regulonDB into a pandas dataframe

import pandas as pd

def data_loading(tsv_file):
    column_names = []
    header_cols = 0
    with open(tsv_file, 'r') as file:
        for line in file:
            header_cols += 1
            if line.startswith("1)"):
                column_names = line.strip().split('\t')
                break
    column_names = [name.split(')')[1] for name in column_names]

    operon_dataframe = pd.read_csv(tsv_file, sep='\t', skiprows=header_cols, header=None, comment="#")

    operon_dataframe.columns = column_names
    return operon_dataframe