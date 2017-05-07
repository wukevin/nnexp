#!/usr/bin/env python3
"""
Classes and functions for interacting with gtf files
"""
import os
import sys
import intervaltree
import collections
import tcga_parser

class Gtf(object):
    def __init__(self, gtf_file, entry_type, entry_levels=set(['ensembl_havana'])):
        # Allows us to query by chromosome and positions
        self.itree = collections.defaultdict(intervaltree.IntervalTree)
        # Allows us to query by gene name
        self.genedict = collections.defaultdict(list)
        # Read in the gtf file
        with open(gtf_file, 'r') as handle:
            for line in handle:
                line = line.rstrip()
                if line[0] == "#": # Skip comments
                    continue
                if line[-1] == ";":  # Make sure we don't end in a ;
                    line = line[:-1]
                # Split the line into its parts
                tokenized = line.split("\t")
                chromosome, confidence, type_of_entry = tokenized[:3]
                if 'chr' not in chromosome: #Make sure the chromosome format is consistent
                    chromosome = "chr" + chromosome
                if (entry_type != 'all' and type_of_entry != entry_type) or confidence not in entry_levels:
                    continue

                start, stop = int(tokenized[3]), int(tokenized[4])
                if stop <= start:
                    continue # Skip these entries

                data = {}
                for chunk in tokenized[-1].split(";"):
                    k, v = chunk.rstrip().split()
                    data[k] = v.replace('"', '')
                # Also give it some additional data
                addtl_data_names = ['chromosome', 'start', 'stop']
                addtl_data = [chromosome, start, stop]
                for name, value in zip(addtl_data_names, addtl_data):
                    assert name not in data # Make sure that we aren't overwriting existing entries
                    data[name] = value

                # Add to the interval tree
                self.itree[chromosome][start:stop] = data
                # Add to the gene dict
                self.genedict[data['gene_name']].append(data)

    def get_overlapping_entries(self, chromosome, position, end_position=None):
        """Given """
        assert isinstance(chromosome, str) and "chr" in chromosome
        assert isinstance(position, int)
        if end_position is not None:
            if not isinstance(end_position, int):
                raise TypeError
            return self.itree[chromosome][position:end_position]
        else:
            return self.itree[chromosome][position]

    def get_gene_entries(self, query_gene):
        """
        Given a query gene name, retrieve all the entries that are associated with it
        If the query gene name is not found, then return None"""
        try:
            return self.genedict[query_gene]
        except KeyError:
            return None


if __name__ == "__main__":
    ensembl_genes = Gtf(os.path.join(tcga_parser.DRIVE_ROOT, "Homo_sapiens.GRCh37.87.gtf"), "gene",
                        set(['ensembl_havana']))
    print(ensembl_genes.get_overlapping_entries('chr1', 11870))
    print(ensembl_genes.get_gene_entries("ERBB2"))
