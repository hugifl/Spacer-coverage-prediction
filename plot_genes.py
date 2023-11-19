import csv
from matplotlib import pyplot
import numpy
import random


def plot_operons(dataframe, genedata, outpath, no_operons):
    if no_operons == 0:
        return "No plots generated"
    
    random_operons_single_gene = []
    while len(random_operons_single_gene)<int(no_operons/2):
        random_operon = dataframe.sample(n=1)
        if random_operon['numberOfGenes'].values[0].max() != 1:
            continue
        if random_operon['Coverage'].values[0].max() > 10:
             random_operons_single_gene.append(random_operon['operonName'].values[0])

    random_operons_multiple_genes = []
    while len(random_operons_multiple_genes)<int(no_operons/2):
        random_operon = dataframe.sample(n=1)
        if random_operon['numberOfGenes'].values[0].max() == 1:
            continue
        if random_operon['Coverage'].values[0].max() > 10:
             random_operons_multiple_genes.append(random_operon['operonName'].values[0])

    random_operons = random_operons_single_gene + random_operons_multiple_genes
    print(len(random_operons))
    for operon in random_operons:
        operon_start = dataframe.loc[dataframe['operonName'] == operon, 'firstGeneLeftPos'].values[0]
        operon_end = dataframe.loc[dataframe['operonName'] == operon, 'lastGeneRightPos'].values[0]
        operon_coverage = dataframe.loc[dataframe['operonName'] == operon, 'Coverage'].values[0]
        operonsize = abs(operon_end - operon_start)
        edge = 0.5 * (len(operon_coverage) - operonsize)

        multiple_genes = int(dataframe.loc[dataframe['operonName'] == operon, 'numberOfGenes'].values[0]) > 1
        # Find the offsets of the start and end sites of the single genes
        if multiple_genes:
            genes = dataframe.loc[dataframe['operonName'] == operon, 'operonGenes'].values[0].split(';')
            positions  = [(genedata[genedata['geneName'].str.contains(gene)]['leftEndPos'].values[0], genedata[genedata['geneName'].str.contains(gene)]['rightEndPos'].values[0]) for gene in genes]
            offsets = [(int(x) - operon_start, int(y) - operon_start) for x, y in positions]
            


        x_coord=numpy.arange(-edge, operonsize + edge)
        
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(12, 6))
        pyplot.plot( x_coord, operon_coverage, color="blue")
        pyplot.title(operon + " No genes: " + str(dataframe.loc[dataframe['operonName'] == operon, 'numberOfGenes'].values[0]))
        pyplot.axvline(x=0, ls="-", lw="2")
        pyplot.axvline(x=operonsize, ls="-", lw="2")
        if multiple_genes:
            for offset_pair in offsets:
                pyplot.axvline(x=offset_pair[0], ls="-.", lw="1",color="blue")
                pyplot.axvline(x=offset_pair[1], ls="-.", lw="1",color="green")
        pyplot.xlabel('OSS = operon start site     Position     OES = operon end site')
        pyplot.ylabel('Raw coverage')
        pyplot.ylim(ymin = 0, ymax=operon_coverage.max()+1) 
        pyplot.xticks([-edge, 0, operonsize/4, operonsize/2, 3*operonsize/4, operonsize,  operonsize +edge], [-edge,'OSS', round(operonsize/4), round(operonsize/2), round(3*operonsize/4), 'OES', edge])
        pyplot.savefig(outpath+"_"+operon+"_operoncoverage.png")
        pyplot.close()
    return "plots generated"

def plot_genes(file,outpath, no_genes):
    if no_genes == 0:
        return "No plots generated"
    data = []
    with open(file, "r") as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            data.append(row)

    data_array = numpy.array(data)
    data_array = data_array[1:,4:]
    coverage_profiles = data_array[:,1:]
    coverage_profiles = coverage_profiles.astype(float)
    for i in range(data_array.shape[0]):       #reverse the genes that are on the negative strand
        if data_array[i, 0] == -1:
            coverage_profiles[i] = coverage_profiles[i, ::-1]
    counter = 0
    max = 0
    random_genes = []
    while len(random_genes)<no_genes:  
        counter += 1
        random_gene = random.randint(0,coverage_profiles.shape[0])
        start = int(0.5 * (len(coverage_profiles[random_gene,:]) - float(data[random_gene][1])))
        end = int(0.5 * (len(coverage_profiles[random_gene,:]) - float(data[random_gene][1])) + float(data[random_gene][1]))
        if coverage_profiles[random_gene,start:end].max() > 10:
             random_genes.append(random_gene)
        if counter > 10000:
             break


    for gene in random_genes:
        exonsize = float(data[gene][1])
        edge = 0.5 * (len(coverage_profiles[0,:]) - exonsize)
        x_coord=numpy.arange(-edge, exonsize + edge)
        start_exon = int(0.5 * (len(coverage_profiles[gene,:]) - float(data[gene][1])))
        start_next_exon = int(0.5 * (len(coverage_profiles[gene+1,:]) - float(data[gene+1][1])))
        start_difference = int(float(data[gene+1][2]) - float(data[gene][2])) 
        x_offset_next_gene_start = start_difference + start_next_exon - start_exon
        x_offset_next_exon_end = x_offset_next_gene_start + int(float(data[gene+1][1]))

        start_prev_exon = int(0.5 * (len(coverage_profiles[gene-1,:]) - float(data[gene-1][1])))
        start_difference_2 = int(float(data[gene][2]) - float(data[gene-1][2])) 
        x_offset_prev_gene_start = start_difference_2 - start_prev_exon + start_exon


        pyplot.style.use('ggplot')
        pyplot.plot( x_coord, coverage_profiles[gene,:], color="blue")
        pyplot.title(data[gene][0])
        pyplot.axvline(x=0, ls="-.", lw="2")
        pyplot.axvline(x=exonsize, ls="-.", lw="2")
        pyplot.axvline(x=x_offset_next_gene_start, ls="-.", lw="2", color='blue')
        pyplot.axvline(x=x_offset_next_exon_end, ls="-.", lw="2", color='blue')
        pyplot.axvline(x=-x_offset_prev_gene_start, ls="-.", lw="2", color='green')
        pyplot.axvline(x=-x_offset_prev_gene_start + int(float(data[gene-1][1])) , ls="-.", lw="2", color='green')
        pyplot.xlabel('ESS = exon start site     Position     EES = exon end site')
        pyplot.ylabel('Scaled coverage')
        pyplot.ylim(ymin = 0, ymax=coverage_profiles[gene,:].max()+1) 
        pyplot.xticks([-edge, -edge/2, 0, exonsize/4, exonsize/2, 3*exonsize/4, exonsize, exonsize + (edge/2), exonsize +edge], [-edge, -round(edge/2),'ESS', round(exonsize/4), round(exonsize/2), round(3*exonsize/4), 'EES', round(edge/2), edge])
        pyplot.savefig(outpath+data[gene][0]+"_exoncoverage2.png")
        pyplot.close()
    return "plots generated"


def plot_genes_scaled(file,outpath, window_size, exonsize, no_genes):
    if no_genes == 0:
        return "No plots generated"
    data = []
    with open(file, "r") as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            data.append(row)

    data_array = numpy.array(data)
    data_array = data_array[1:,4:]
    coverage_profiles = data_array[:,1:]
    coverage_profiles = coverage_profiles.astype(float)
    for i in range(data_array.shape[0]):       #reverse the genes that are on the negative strand
        if data_array[i, 0] == -1:
            coverage_profiles[i] = coverage_profiles[i, ::-1]
    counter = 0
    max = 0
    random_genes = []
    while len(random_genes)<no_genes:  
        counter += 1
        random_gene = random.randint(0,coverage_profiles.shape[0])
        if coverage_profiles[random_gene,:].max() > 10:
             random_genes.append(random_gene)
        if counter > 10000:
             break

    x_coord=numpy.arange( -window_size, exonsize+window_size) # user defined or default size of coverage plot
    x_coord_smooth = numpy.linspace( -window_size, exonsize+window_size - 1,300) # x coordinates used for smoothing
    # adding exon start and end to smoothing coordinates
    if 0 not in x_coord_smooth:
    	x_coord_smooth=numpy.append(x_coord_smooth, [0], axis=0)
   
    if exonsize not in x_coord_smooth:
    	x_coord_smooth=numpy.append(x_coord_smooth, [exonsize], axis=0)
    x_coord_smooth = numpy.sort(x_coord_smooth)

    
    for gene in random_genes:
        pyplot.style.use('ggplot')
        pyplot.plot( x_coord, coverage_profiles[gene,:], color="blue")
        pyplot.title(data[gene][0])
        pyplot.axvline(x=0, ls="-.", lw="2")
        pyplot.axvline(x=exonsize, ls="-.", lw="2")
        pyplot.xlabel('ESS = exon start site     Position     EES = exon end site')
        pyplot.ylabel('Scaled coverage')
        pyplot.ylim(ymin = 0, ymax=coverage_profiles[gene,:].max()+1) 
        pyplot.xticks([-window_size, -window_size/2, 0, exonsize/4, exonsize/2, 3*exonsize/4, exonsize, exonsize + (window_size/2), exonsize +window_size], [-window_size, -round(window_size/2),'ESS', '25%', '50%', '75%', 'EES', round(window_size/2), window_size])
        pyplot.savefig(outpath+data[gene][0]+"_exoncoverage.png")
        pyplot.close()
    return "plots generated"
    

