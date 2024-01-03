import pandas as pd

def load_txt_to_dataframe(file_path):
    # Load the text file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t')  # assuming tab-separated values; modify if different
    return df

def print_non_int_end_rows(df):
    # Iterate over the DataFrame and try to convert 'End' values to int
    non_int_rows = []
    for index, row in df.iterrows():
        try:
            int(row['End'])
        except ValueError:
            # If conversion fails, add the row to the list
            non_int_rows.append(row)
            print(row)

    # Convert the list to a DataFrame and print it
    non_int_df = pd.DataFrame(non_int_rows)
    print(non_int_df)

# Example usage
file_path = '/cluster/home/hugifl/spacer_coverage_input/genomeCounts_paraquat.txt'  # replace with your file path
df = load_txt_to_dataframe(file_path)
print_non_int_end_rows(df)

