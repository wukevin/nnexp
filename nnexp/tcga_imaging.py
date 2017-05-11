"""
Functions and classes for creating images that are expressive of
per-patient expression profiles

Notes to self:
   * Convert gene names into the intervaltree, much like how the cnvs
   are represented. This allows us to preserve the level of detail
   granted by CNV data, and combine it with the more general information
   given by rna/protein expression data
   * Another idea is to use every single data point that we have observations
   for, i.e a union of the cnv, rna, and protein expression data. we'd then
   feed these images into the neural net and let that decide for itself what is
   important. If we do this, we need to make sure that we are ordering pixels
   by their genomic coordinates such that areas that are close to each other
   end up getting grouped together, simulating the idea of an amplicon
"""

import os
import glob
import pickle
import intervaltree
import sortedcontainers
import collections
import gtf_parser
import tcga_parser
from PIL import Image


##### HERE ARE ALL THE IMAGE CREATORS #####


def create_image_full_gene_intersection(patient):
    """
    Consumes a tcga patient and creates an image that is a composite of
    only genes/positions that are completely within
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)

    # Only retain genes that are common to both genes and protein expression
    # data
    common_genes = set(patient.gene_exp.keys()).intersection(set(patient.prot_exp.keys()))
    print(len(common_genes)) # This is only 44...we may want to try something else


def create_image_full_union(patient, gene_intervals):
    """
    Another idea is to use every single data point that we have observations
    for, i.e a union of the cnv, rna, and protein expression data. we'd then
    feed these images into the neural net and let that decide for itself what is
    important. If we do this, we need to make sure that we are ordering pixels
    by their genomic coordinates such that areas that are close to each other
    end up getting grouped together, simulating the idea of an amplicon
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)
    assert isinstance(gene_intervals, gtf_parser.Gtf)

    # reads in the data as both interval trees and in sorted dicts
    rna_intervals = gene_to_interval(patient.gene_values(), gene_intervals)
    rna_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in rna_intervals.items()}
    protein_intervals = gene_to_interval(patient.prot_values(), gene_intervals)
    protein_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in protein_intervals.items()}
    cnv_intervals = patient.cnv_values()
    cnv_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in cnv_intervals.items()}
    # We want to split the sorted intervals such that they all break at the same places
    # and they stack up neatly on one another

    # Count how many points are covered in the total union
    # counters = collections.defaultdict(int)
    # for chromosome, itree in total_union.items():
    #     for i in range(itree.begin(), itree.end()):
    #         if itree.overlaps(i):
    #             counters[chromosome] += 1
    #     print("%s\t%s\t%i" % (patient.barcode, chromosome, counters[chromosome]))
    # print("%s\t%i" % (patient.barcode, sum(counters.values())))

    # union_intervals = patient.cnv.union(rna_intervals)
    # union_intervals = union_intervals.union(protein_intervals)

    # total_size = 0
    # for i in range(union_intervals.start(), unionintervals.end()):
    #     if union_intervals[]
##### HERE ARE THE UTILITY FUNCTIONS FOR THEM #####


def gene_to_interval(gene_dict, gtf, verbose=False):
    """
    Converts a dictionary that maps genes to chromosomes to entries to a
    intervaltree of entries. Things that have no corresponding intervals
    associated in the given gtf are left out. The gtf should be read in for
    genes, since that is what we are querying (e.g. not exons)
    """
    assert isinstance(gtf, gtf_parser.Gtf)
    itree = collections.defaultdict(intervaltree.IntervalTree)

    counter = 0
    for key, value in gene_dict.items():
        gtf_entries = gtf.get_gene_entries(key)
        if len(gtf_entries) == 0:
            continue
        if len(gtf_entries) > 1:
            # There was more than one. pick the one with the highest gene version
            entry_versions = [int(x['gene_version']) for x in gtf_entries]
            highest_version = max(entry_versions)
            accept_indicies = [x for x, y in enumerate(entry_versions) if y == highest_version]
            if len(accept_indicies) > 1:
                continue
            elif len(accept_indicies) == 0:
                raise RuntimeError("Could not identify a highest version")
            assert len(accept_indicies) == 1
            gtf_entry = gtf_entries[accept_indicies[0]]
        else:
            gtf_entry = gtf_entries[0]

        itree[gtf_entry['chromosome']][gtf_entry['start']:gtf_entry['stop']] = value
        counter += 1
    if verbose:
        print("Converted %i/%i entries from gene dict to intervaltree" % (counter, len(gene_dict)))
    return itree

def interval_to_sorteddict(interval_tree):
    """
    Converts an interval tree to a sorted dictionary where the keys of the dictionary
    are tuples made up of each interval's start and stop positions. This allows us to
    traverse the data in the interval tree in sorted order, instead of the arbitrary
    order that the interval tree returns in its iterator
    """
    assert isinstance(interval_tree, intervaltree.IntervalTree)
    converted = sortedcontainers.SortedDict()
    for interval in interval_tree:
        converted[(interval.begin, interval.end)] = interval.data
    return converted


def main():
    """Runs the script"""
    # Load in a dummy file
    pattern = os.path.join(
        tcga_parser.DATA_ROOT,
        "tcga_patient_objects",
        "TCGA*.pickled"
    )
    tcga_patient_files = glob.glob(pattern)
    ensembl_genes = gtf_parser.Gtf(os.path.join(tcga_parser.DRIVE_ROOT, "Homo_sapiens.GRCh37.87.gtf"),
                                   "gene", set(['ensembl_havana']))
    if len(tcga_patient_files) == 0:
        raise RuntimeError("Found no files matching pattern:\n%s" % pattern)

    for patient_file in tcga_patient_files:
        with open(patient_file, 'rb') as handle:
            patient = pickle.load(handle)
        assert isinstance(patient, tcga_parser.TcgaPatient)
        create_image_full_union(patient, ensembl_genes)

if __name__ == "__main__":
    main()
