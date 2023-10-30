### Florian Hugi, adapted from Tanmay Tanna ### 26-10-2023 ###
# dimension of coverage vector is the same for each gene and corresponds to the length neceessary for the upstream-exon-downstream of the longest exon
from __future__ import division
import HTSeq
import numpy
from matplotlib import pyplot
import argparse
from scipy.interpolate import interp1d
import csv
import pandas as pd
import math
from plot_genes import plot_genes

parser = argparse.ArgumentParser()

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')  # change grouping of argument to required or optional in help through this chunk of code

### inputs ###

# required 
required.add_argument('-i','--inbamlist',  nargs='+', help='array of bam files with path', required=True)
required.add_argument('-g', '--genome', help='genome annotation file path (gff/gtf)', required=True)


# optional
optional.add_argument('-o', '--outPath', help='path to output directory.', default='.')
optional.add_argument('-w', '--winwidth', help='distance upstream and downstream of exon to be displayed Default=300 bp', type=int, dest='winwidth', default=300)
optional.add_argument('-plots', '--plots', help='number of example gene plots to generate', type=int, dest='plots', default=0)

#initialize values

parser._action_groups.append(optional) 
args = parser.parse_args()

bamlist=args.inbamlist
gtffile = HTSeq.GFF_Reader(str(args.genome))
outdir=str(args.outPath)
winwidth = int(args.winwidth)
plots = int(args.plots)
fragmentsize = 200 # This sets size of alignment to 200 as 200 bp is more representative of standard fragment length than the observed readlength 




filenumber=0 # file counter

# allocating space and initializing gene coverage dictionary:
max_exon_length = 0
for feature in gtffile:
	if feature.type == "gene":
		exon_length = abs(feature.iv.end_d_as_pos.pos - feature.iv.start_d_as_pos.pos)
		if exon_length > max_exon_length:
			max_exon_length = exon_length
	
N_BAM = len(bamlist)
exonpos = list() # list to store the start and end positions of a gene
coverage_dict = {}
for feature in gtffile:
	if feature.type == "gene":
		exon_length = abs(feature.iv.end_d_as_pos.pos - feature.iv.start_d_as_pos.pos)
		if exon_length < 100:
			continue

		start_pos_exon_left_to_right = min(feature.iv.end_d_as_pos.pos, feature.iv.start_d_as_pos.pos)
		region_kept_to_the_left_of_exon = (winwidth + math.floor(0.5 * (max_exon_length - exon_length))) + 1
		end_pos_exon_left_to_right = max(feature.iv.end_d_as_pos.pos, feature.iv.start_d_as_pos.pos)
		region_kept_to_the_right_of_exon = (winwidth + math.ceil(0.5 * (max_exon_length - exon_length))) + 1
		chromosome_length = 5000000
		if ((start_pos_exon_left_to_right - region_kept_to_the_left_of_exon)<0) or ((end_pos_exon_left_to_right + region_kept_to_the_right_of_exon)>chromosome_length): #we need a consistent length upstream-exon-downstream for all genes. exclude genes that are too close to the start or the end of the chromosome
			continue
		exonpos.append( [feature.iv.start_d_as_pos, feature.iv.end_d_as_pos] ) 
		gene_name = feature.name
		coverage_dict[gene_name] = numpy.zeros((N_BAM, ((max_exon_length + 2* winwidth))+4), dtype='f') # + 3 to also store position info (direction, start, end) #exon_length + 2*...
		coverage_dict[gene_name][:,0] = exon_length

gene_names = list(coverage_dict.keys())
### looping through for each bam file ###
for j,file in enumerate(bamlist):
	filenumber+=1
	bamfile = HTSeq.BAM_Reader(str(file))
	coverage = HTSeq.GenomicArray( "auto", stranded=False, typecode="i" ) #initializing coverage array (array like datastructure)
	#outfile= file.split("/")[-1]

	readnumber=0 # number of reads

	for almnt in bamfile:
		readnumber+=1
		if almnt.aligned:
			if almnt.iv.end > fragmentsize:
				almnt.iv.length = fragmentsize
			else:
				almnt.iv.length = almnt.iv.end
			coverage[ almnt.iv ] += 1  # add 1 to coverage for each aligned "fragment"

	#profile = numpy.zeros( 2*winwidth + exonsize , dtype='f' ) # average local (for current bam) coverage profile

	for i, p in enumerate(exonpos):
		exon_length = int(coverage_dict[gene_names[i]][j,0])
		
		if p[0].pos<p[1].pos: # positive strand
			exon = HTSeq.GenomicInterval( p[0].chrom, p[0].pos , p[1].pos , "." )
			upstream = HTSeq.GenomicInterval( p[0].chrom, p[0].pos - (winwidth + math.floor(0.5 * (max_exon_length - exon_length))), p[0].pos, "." ) # p[0].chrom, p[0].pos - winwidth, p[0].pos, "."
			downstream = HTSeq.GenomicInterval( p[0].chrom, p[1].pos, p[1].pos + (winwidth + math.ceil(0.5 * (max_exon_length - exon_length))), "." ) # p[0].chrom, p[1].pos, p[1].pos + winwidth, "."
			coverage_dict[gene_names[i]][j,1] = p[0].pos - (winwidth + math.floor(0.5 * (max_exon_length - exon_length)))
			coverage_dict[gene_names[i]][j,2] = p[1].pos + (winwidth + math.ceil(0.5 * (max_exon_length - exon_length)))
			coverage_dict[gene_names[i]][j,3] = 1 #marker for positive strand
		else: # negative strand I changed upstream and downstream here!! so that it is relative to the gene direction
			exon = HTSeq.GenomicInterval( p[0].chrom, p[1].pos+1 , p[0].pos+1 , "." )
			downstream = HTSeq.GenomicInterval( p[0].chrom, p[1].pos - (winwidth + math.ceil(0.5 * (max_exon_length - exon_length))) + 1, p[1].pos+1, "." ) # p[0].chrom, p[1].pos - winwidth+1, p[1].pos+1, "."
			upstream = HTSeq.GenomicInterval( p[0].chrom, p[0].pos+1, p[0].pos + (winwidth + math.floor(0.5 * (max_exon_length - exon_length))) + 1,  "." ) # p[0].chrom, p[0].pos+1, p[0].pos + winwidth +1,  "."
			coverage_dict[gene_names[i]][j,1] = p[1].pos - (winwidth + math.ceil(0.5 * (max_exon_length - exon_length))) + 1
			coverage_dict[gene_names[i]][j,2] = p[0].pos + (winwidth + math.floor(0.5 * (max_exon_length - exon_length))) + 1
			coverage_dict[gene_names[i]][j,3] = -1 #marker for negative strand
		
		exoncvg = numpy.fromiter( coverage[exon], dtype='f', count = exon_length) # coverage over exon
	    #exoncvg_scaled = numpy.zeros( exonsize, dtype='f' ) # scaling coverage over each exon to user defined or default exonsize for comparability
	    #k=(k-1)/exonsize # increment to be used for loop #k is not exon_length, would need to change that here
	    #i=0 # loop iterators
	    #j=0
	    #while j<exonsize:
	    #    if round(i)!=round(i+k):
		#        exoncvg_scaled[j]=numpy.mean(exoncvg[int(round(i)):int(round(i+k))])
	    #    else:
		#        exoncvg_scaled[j]=exoncvg[int(round(i))]
	    #    i+=k
	    #    j+=1
	    ## concatenating coverage over upstream and downstream window with scaled exon coverage
	    #wincvg = numpy.concatenate((numpy.fromiter(coverage[upstream], dtype='f', count=winwidth), exoncvg_scaled), axis = 0)  
	    #wincvg = numpy.concatenate((wincvg, numpy.fromiter(coverage[downstream], dtype='f', count=winwidth)), axis = 0)
		if p[0].pos<p[1].pos:
			up_wincvg = numpy.fromiter(coverage[upstream], dtype='f', count = int((winwidth + math.floor(0.5 * (max_exon_length - exon_length)))))
			wincvg = numpy.concatenate((up_wincvg, exoncvg), axis = 0)  
			wincvg = numpy.concatenate((wincvg, numpy.fromiter(coverage[downstream], dtype='f', count= int((winwidth + math.ceil(0.5 * (max_exon_length - exon_length)))))), axis = 0)
		else:
			wincvg = numpy.concatenate((numpy.fromiter(coverage[downstream], dtype='f', count= int((winwidth + math.ceil(0.5 * (max_exon_length - exon_length))))), exoncvg), axis = 0)  
			wincvg = numpy.concatenate((wincvg, numpy.fromiter(coverage[upstream], dtype='f', count= int((winwidth + math.floor(0.5 * (max_exon_length - exon_length)))))), axis = 0)
		
		coverage_dict[gene_names[i]][j,4:] = wincvg #add the coverage for gene i to the row in the matrix corresponding to the j-th bam file
	   
combined_list = []

for key in coverage_dict:
	gene_list = coverage_dict[key].tolist()
	gene_list = [[key] + row for row in gene_list]
	if not combined_list:
		combined_list = gene_list
		
	else:
		combined_list = combined_list + gene_list

#df = pd.DataFrame(combined_array)
#df.to_csv(outdir+"/gene_coverage.csv", index=False)


column_names = ['Gene_name','Exon_length','Start_pos','End_pos','Strand'] + [f'position_{i}_coverage' for i in range(1, len(combined_list[0]) - 4)]
combined_list = [column_names] + combined_list

# Define the CSV file path
csv_file = outdir+"/gene_coverage_"+str(winwidth)+".csv"

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the data with custom column names
    writer.writerows(combined_list)




print(plot_genes(csv_file,outdir, plots))