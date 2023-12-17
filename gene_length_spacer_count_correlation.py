import pandas as pd
import matplotlib.pyplot as plt

count_file = "/cluster/home/hugifl/spacer_coverage_input/genomeCounts_extended_UMI_gut.txt"
outpath = "/cluster/home/hugifl/spacer_coverage_output/"

df = pd.read_csv(count_file, sep='\t', dtype=str, low_memory=False)

columns_to_convert = ['Start', 'End']  
sample_columns = [col for col in df.columns if col.startswith('BSSE')]

# Convert these columns to integers
for column in sample_columns:
    df[column] = df[column].astype(int)

# Convert the specified columns to integers
for column in columns_to_convert:
    df[column] = df[column].astype(int)

# Calculate gene length
df['Gene_Length'] = (df['End'] - df['Start']).abs()

# Sum the read counts across all samples
sample_columns = [col for col in df.columns if col.startswith('BSSE')]
df['Total_Read_Count'] = df[sample_columns].sum(axis=1)

# Plotting
plt.scatter(df['Gene_Length'], df['Total_Read_Count'])
plt.xlabel('Gene Length (log(bp))')
plt.ylabel('Total Spacer Count (log(count))')
plt.yscale('log')
plt.xscale('log')
plt.title('Correlation between Gene Length and Total Read Count')
plt.savefig(outpath + "GeneLength_SpacerCount_correlation_gut")