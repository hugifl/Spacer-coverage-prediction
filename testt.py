import HTSeq
import csv

gtf_file = HTSeq.GFF_Reader("/cluster/home/hugifl/recordseq-workflow-dev/dev-hugi/exon_coverage/U00096.3.gff3")  # Replace with the path to your GTF file
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

a = []
for i in range(4):
    a.append(i)
print(a)

