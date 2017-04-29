#!/usr/bin/env python3
"""
Classes and functions for interacting with gtf files
"""

import sys
import intervaltree
import collections

class Gtf(object):
    def __init__(self, gtf_file, entry_type, entry_levels=set(['ensembl_havana'])):
        # Allows us to query by chromosome and positions
        self.itree = collections.defaultdict(intervaltree.IntervalTree)
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

                self.itree[chromosome][start:stop] = data

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


if __name__ == "__main__":
    ensembl_genes = Gtf("/Volumes/Data/Homo_sapiens.GRCh37.87.gtf/Homo_sapiens.GRCh37.87.gtf", "gene",
                        set(['ensembl_havana', 'ensembl', 'havana']))
    print(ensembl_genes.get_overlapping_entries('chr1', 11870))
