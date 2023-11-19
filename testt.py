
import csv

tsv_file = "/cluster/home/hugifl/exon_coverage/OperonSet.tsv"
genome_file = "/cluster/home/hugifl/exon_coverage_input_output/U00096.3.fasta"
outfile = "/cluster/home/hugifl/recordseq-workflow-dev/dev-hugi/exon_coverage/output/gene_coverage_scaled.csv"
# Open the GTF file for reading
#counter = 0
#maximum = 0
#for a in gtf_file:
#    end = max(a.iv.start_d_as_pos.pos, a.iv.end_d_as_pos.pos)
#    if end > maximum:
#        maximum = end
#
#print(maximum)
import pandas as pd

# Define the filename

count_df = pd.read_csv('genomeCounts_extended_UMI.txt', sep='\t', dtype=str, low_memory=False)

df = pd.read_csv('exon_coverage_input_output/output/window_coverage_data.csv', sep=',', dtype=str, low_memory=False)
for col in df.columns[3:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop NaN values to ensure only numeric data is considered
numeric_df = df.iloc[:, 3:].dropna()

# Flatten the DataFrame and sort values
flattened_series = numeric_df.stack()
sorted_values = flattened_series.sort_values(ascending=False)

# Select the top 10 values
top_10_values = sorted_values.head(10)

print(top_10_values)