### Florian Hugi, adapted from Tanmay Tanna ### 31-10-2023 ###
# dimension of coverage vector is the different for each operon and corresponds to the length upstream-operon-downstream 

from __future__ import division
import HTSeq
import numpy
from matplotlib import pyplot
import argparse
from scipy.interpolate import interp1d
import csv
import pandas as pd
import math
from plot_genes import plot_operons
from data_loading import data_loading

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-i','--inbamlist',  nargs='+', help='array of bam files with path', required=True)
required.add_argument('-op', '--operons', help='genome annotation file path (regulonDB tsv)', required=True)
required.add_argument('-ge', '--genes', help='genome annotation file path (regulonDB tsv)', required=True)

# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='distance upstream and downstream of exon to be displayed Default=50 bp', type=int, dest='winwidth', default=50)
optional.add_argument('-plots', '--plots', help='number of example gene plots to generate', type=int, dest='plots', default=0)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

bamlist=args.inbamlist
tsv_file = str(args.operons)
tsv_file_2 = str(args.genes)
outdir=str(args.outPath)
winwidth = int(args.winwidth)
plots = int(args.plots)
fragmentsize = 200 # This sets size of alignment to 200 as 200 bp is more representative of standard fragment length than the observed readlength 

############ loading operon and gene information data ############
operon_dataframe = data_loading(tsv_file)
gene_dataframe = data_loading(tsv_file_2)

# allocating space and initializing gene coverage dictionary:
N_BAM = len(bamlist)
coverage_df = operon_dataframe
#coverage_df['Coverage'] = 0 #[[]] * coverage_df.shape[0]
coverage_df['OperonLength'] = 0

def set_empty_list(row):
    row['Coverage'] = []
    return row

# Apply the function to each row
coverage_df = coverage_df.apply(set_empty_list, axis=1)


for index, row in coverage_df.iterrows():
	operon_length = abs(row['firstGeneLeftPos'] - row['lastGeneRightPos'])
	coverage_df.loc[index, 'OperonLength'] = operon_length
	region_kept_on_the_edge_of_operon = winwidth
	if (row['firstGeneLeftPos']-region_kept_on_the_edge_of_operon)< 0:
		coverage_df = coverage_df.drop(index)
		continue
	if operon_length < 100:
		coverage_df = coverage_df.drop(index)
		continue
	if row['firstGeneLeftPos'] > row['lastGeneRightPos']:
		coverage_df = coverage_df.drop(index)
		continue
	
### looping through for each bam file ###
for j,file in enumerate(bamlist):
	bamfile = HTSeq.BAM_Reader(str(file))
	coverage = HTSeq.GenomicArray( "auto", stranded=False, typecode="i" ) #initializing coverage array (array like datastructure)

	for almnt in bamfile:
		if almnt.aligned:
			#if almnt.iv.end > fragmentsize:
			#	almnt.iv.length = fragmentsize
			#else:
			#	almnt.iv.length = almnt.iv.end
			coverage[almnt.iv] += 1  # add 1 to coverage for each aligned "fragment"

	for index, row in coverage_df.iterrows():
		operon_length = coverage_df.loc[index, 'OperonLength']
		
		if coverage_df.loc[index, 'strand'] == "forward": # positive strand
			operon = HTSeq.GenomicInterval( "U00096.3", coverage_df.loc[index, 'firstGeneLeftPos'] , coverage_df.loc[index, 'lastGeneRightPos'] , ".") 
			upstream = HTSeq.GenomicInterval( "U00096.3", coverage_df.loc[index, 'firstGeneLeftPos'] - (winwidth), coverage_df.loc[index, 'firstGeneLeftPos'], "." )
			downstream = HTSeq.GenomicInterval( "U00096.3", coverage_df.loc[index, 'lastGeneRightPos'], coverage_df.loc[index, 'lastGeneRightPos'] + (winwidth), "." )
			
		else: # negative strand I changed upstream and downstream here!! so that it is relative to the gene direction
			operon = HTSeq.GenomicInterval( "U00096.3", coverage_df.loc[index, 'firstGeneLeftPos'] , coverage_df.loc[index, 'lastGeneRightPos'] , ".") 
			upstream = HTSeq.GenomicInterval( "U00096.3", coverage_df.loc[index, 'lastGeneRightPos'], coverage_df.loc[index, 'lastGeneRightPos'] + (winwidth), "."  )
			downstream = HTSeq.GenomicInterval( "U00096.3", coverage_df.loc[index, 'firstGeneLeftPos'] - (winwidth), coverage_df.loc[index, 'firstGeneLeftPos'], "." )
			
		
		operoncvg = numpy.fromiter(coverage[operon], dtype='f', count = operon_length) 
	    
		if coverage_df.loc[index, 'strand'] == "forward":
			wincvg = numpy.concatenate((numpy.fromiter(coverage[upstream], dtype='f', count = int((winwidth))), operoncvg), axis = 0)  
			wincvg = numpy.concatenate((wincvg, numpy.fromiter(coverage[downstream], dtype='f', count= int((winwidth)))), axis = 0)
		else:
			wincvg = numpy.concatenate((numpy.fromiter(coverage[downstream], dtype='f', count= int((winwidth))), operoncvg), axis = 0)  
			wincvg = numpy.concatenate((wincvg, numpy.fromiter(coverage[upstream], dtype='f', count= int((winwidth)))), axis = 0)
			wincvg = wincvg[::-1]
		
		coverage_df.loc[index, 'Coverage'].append(wincvg) #add the coverage for gene i to the row in the matrix corresponding to the j-th bam file
		
			
		

############## Saving output ##############
expanded_rows = []
# Iterate through the DataFrame
for index, row in coverage_df.iterrows():
    # Get the last column value (list of lists)
    coverage_profiles = row['Coverage']
    # Iterate through the inner lists
    for profile in coverage_profiles:
        # Create a new row by appending the sublist to the original row
        new_row = row.copy()
        new_row['Coverage'] = profile
        expanded_rows.append(new_row)

# Create a new DataFrame from the expanded rows
expanded_df = pd.DataFrame(expanded_rows)
print("part 1 done ")
#csv_file = outdir+"/operon_coverage._"+str(winwidth)+"_"+".csv"
#expanded_df.to_csv(csv_file, index=False)


############## Plotting ##############
print(plot_operons(expanded_df,gene_dataframe,outdir, plots))