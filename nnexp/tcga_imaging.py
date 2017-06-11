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
   * Conver gene expression data to a logarithmic scale. This should address the
   fact that the gene expression data is on a completely different order of
   magnitude than the CNV or RNASeq expression data
"""

import os
import glob
import pickle
import itertools
import intervaltree
import sortedcontainers
import collections
import gtf_parser
import tcga_parser
from PIL import Image
import numpy as np
import scipy.misc
import time
import math
import functools
import multiprocessing
import tensorflow as tf

import tcga_analysis

IMAGES_DIR = os.path.join(
    tcga_parser.DRIVE_ROOT,
    "results",
    "images"
)
TENSORS_DIR = os.path.join(
    tcga_parser.DRIVE_ROOT,
    "results",
    "tensors"
)

##### HERE ARE ALL THE IMAGE CREATORS #####

def value_within_range(value, minimum, maximum):
    """
    Checks whether the given value falls wtihin the given minimum/maximum range, inclusive
    """
    if math.isclose(value, minimum, rel_tol=1e-6) or math.isclose(value, maximum, rel_tol=1e-6):
        return True
    if value >= minimum and value <= maximum:
        return True
    return False

def create_image_full_gene_intersection(patient):
    """
    Consumes a tcga patient and creates an image that is a composite of
    only genes/positions that are completely within the domain covered by CNV and RNA
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)

    # Only retain genes that are common to both genes and protein expression
    # data
    # common_genes = set(patient.gene_exp.keys()).intersection(set(patient.prot_exp.keys()))
    common_genes = set(patient.gene_exp.keys())
    print(len(common_genes)) # This is only 44...we may want to try something else


def create_image_full_union(patient, gene_intervals, breakpoints_file, ranges_file):
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

    start_time = time.time()
    # reads in the data as both interval trees and in sorted dicts
    try:
        rna_intervals = gene_to_interval(patient.gene_values(), gene_intervals)
        rna_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in rna_intervals.items()}
        protein_intervals = gene_to_interval(patient.prot_values(), gene_intervals)
        protein_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in protein_intervals.items()}
        cnv_intervals = patient.cnv_values()
        cnv_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in cnv_intervals.items()}
    except AttributeError:
        # Some data may be missing for some patients - for those we simply dont' generate an image
        return None

    # # Read in the ranges and breakpoitns file
    breakpoints = {}
    with open(breakpoints_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            chromosome, points = line.split(": ")
            breakpoints[chromosome] = sortedcontainers.SortedList([int(x) for x in points.split(",")])
    ranges = {}
    with open(ranges_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            datatype, minimum, _null, maximum = line.split()
            ranges[datatype] = (float(minimum), float(maximum))

    # In total we have 328934 breakpoints, which comes out to just under 600 * 600
    # https://stackoverflow.com/questions/12062920/how-do-i-create-an-image-in-pil-using-a-list-of-rgb-tuples
    # Create the template for the image that we are going to create
    width = max([len(x) for x in breakpoints.values()])
    height = len(breakpoints.keys())
    channels = 3 # RGB
    img = np.zeros((height, width, channels), dtype=np.uint8) # unsigned 8-bit integers are 0-255
    # We walk through the chromosomes
    for channel_index, sorted_intervals in enumerate([cnv_intervals_sorted, rna_intervals_sorted, protein_intervals_sorted]):
        for row_index, chromosome in enumerate(breakpoints.keys()):
            # Fill in CNV data on a scale of 0-255
            try:
                values_for_chromosome = sorted_intervals[chromosome]
            except KeyError:
                # This chromosome doesn't exist for this datatype - oh well move on
                continue
            for start_stop_tuple, value in values_for_chromosome.items():
                # Normalize the value to be within the 0-255 range
                if channel_index == 0:
                    minimum, maximum = ranges['cnv']
                elif channel_index == 1:
                    minimum, maximum = ranges['gene']
                elif channel_index == 2:
                    minimum, maximum = ranges['prot']
                else:
                    raise ValueError("Unrecognized channel index: %i" % channel_index)
                if not value_within_range(value, minimum, maximum):
                    raise ValueError("%s WARNING: Given value %f does not fall in the min/max range: %f/%f" % (patient.barcode, value, minimum, maximum))
                value_normalized = np.uint8((float(value) - float(minimum)) / float(maximum) * 255)
                
                start, stop = start_stop_tuple # Figure out the positions where it starts/stops
                start_index = breakpoints[chromosome].index(start) # Figure out where those coordinates lie on the list of sorted breakpoints
                stop_index = breakpoints[chromosome].index(stop)
                if not stop_index > start_index:
                    print("%s WARNING: Start index (%i) is not less than stop index (%i) for channel %i, chromosome %s" % (patient.barcode, start, stop, channel_index, chromosome))
                for col_index in range(start_index, stop_index + 1): # +1 to be inclusive of the stop index
                    img[row_index][col_index][channel_index] = value_normalized

    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    image_path = os.path.join(IMAGES_DIR, "%s.expression.png" % patient.barcode)
    scipy.misc.imsave(image_path, img)
    print("Generated %s in %f seconds" % (image_path, time.time() - start_time))


def create_image_full_union_single_vector(patient, gene_intervals, breakpoints_file, ranges_file):
    """
    Uses every single value in the union of all the datatypes, much like the above function.
    The difference is that in the above, we are compositing into a 2D image where the vertical
    axis is thecchromosome, and the horizontal axis is position along that chromosome. Here, all
    the chromosomes sit along the same 1D axis, in alphabetical order.

    For some reason, the png fiels produced by thsi fucntion can't be opened by windows photo viewer
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)
    assert isinstance(gene_intervals, gtf_parser.Gtf)

    start_time = time.time()
    # reads in the data as both interval trees and in sorted dicts
    try:
        rna_intervals = gene_to_interval(patient.gene_values(), gene_intervals)
        rna_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in rna_intervals.items()}
        protein_intervals = gene_to_interval(patient.prot_values(), gene_intervals)
        protein_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in protein_intervals.items()}
        cnv_intervals = patient.cnv_values()
        cnv_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in cnv_intervals.items()}
    except AttributeError:
        # Some data may be missing for some patients - for those we simply dont' generate an image
        return None

    # # Read in the ranges and breakpoints file
    breakpoints = {}
    with open(breakpoints_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            chromosome, points = line.split(": ")
            breakpoints[chromosome] = sortedcontainers.SortedList([int(x) for x in points.split(",")])
    ranges = {}
    with open(ranges_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            datatype, minimum, _null, maximum = line.split()
            ranges[datatype] = (float(minimum), float(maximum))

    # In total we have 328934 breakpoints
    # https://stackoverflow.com/questions/12062920/how-do-i-create-an-image-in-pil-using-a-list-of-rgb-tuples
    width = sum([len(x) for x in breakpoints.values()])
    chr_cum_sum = {} # Describes how many breakpoints occured before the chromosome, noninclusive
    for chromosome in breakpoints.keys():
        chr_cum_sum[chromosome] = sum([len(bps) for chrom, bps in breakpoints.items() if chrom < chromosome])
    # Create the template for the image that we are going to create
    height = 1
    channels = 3 # RGB
    dimensions = (height, width, channels)
    print(dimensions)
    img = np.zeros(dimensions, dtype=np.uint8) # unsigned 8-bit integers are 0-255
    # We walk through the chromosomes
    for channel_index, sorted_intervals in enumerate([cnv_intervals_sorted, rna_intervals_sorted, protein_intervals_sorted]):
        for chromosome in breakpoints.keys():
            # Fill in CNV data on a scale of 0-255
            try:
                values_for_chromosome = sorted_intervals[chromosome]
            except KeyError:
                # This chromosome doesn't exist for this datatype - oh well let's move on
                continue
            for start_stop_tuple, value in values_for_chromosome.items():
                # Normalize the value to be within the 0-255 range
                if channel_index == 0:
                    minimum, maximum = ranges['cnv']
                elif channel_index == 1:
                    minimum, maximum = ranges['gene']
                elif channel_index == 2:
                    minimum, maximum = ranges['prot']
                else:
                    raise ValueError("Unrecognized channel index: %i" % channel_index)
                if not value_within_range(value, minimum, maximum):
                    raise ValueError("%s WARNING: Given value %f does not fall in the min/max range: %f/%f" % (patient.barcode, value, minimum, maximum))
                value_normalized = np.uint8((float(value) - float(minimum)) / float(maximum) * 255)
                
                start, stop = start_stop_tuple # Figure out the positions where it starts/stops
                start_index = breakpoints[chromosome].index(start) # Figure out where those coordinates lie on the list of sorted breakpoints
                stop_index = breakpoints[chromosome].index(stop)
                if not stop_index > start_index:
                    print("%s WARNING: Start index (%i) is not less than stop index (%i) for channel %i, chromosome %s" % (patient.barcode, start, stop, channel_index, chromosome))
                for col_index in range(start_index, stop_index + 1): # +1 to be inclusive of the stop index
                    col_index_with_offset = col_index + chr_cum_sum[chromosome]
                    assert col_index_with_offset < width
                    for i in range(height):
                        img[i][col_index_with_offset][channel_index] = value_normalized
    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
    image_path = os.path.join(IMAGES_DIR, "%s.expression.png" % patient.barcode)
    scipy.misc.imsave(image_path, img)
    print(img.shape)
    print("Generated %s in %f seconds" % (image_path, time.time() - start_time))


def create_gene_intersection_single_vector(patient, gene_intervals, genes_file, ranges_file):
    """
    Creates vectors of two rows, n columns where each column is a gene and each row is a datatype.
    Datatypes include CNV data and RNA data. RNA data is log-transformed. Both CNV and RNA values
    are reported as a percentage of their maximum value.
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)
    assert isinstance(gene_intervals, gtf_parser.Gtf)

    start_time = time.time()
    # Load in the genes we'll be using and make sure they're of consistent ordering
    genes = sortedcontainers.SortedSet()
    with open(genes_file, 'r') as handle:
        for gene in handle:
            gene = gene.rstrip()
            genes.add(gene)
    # Load in the ranges file
    ranges = {}
    with open(ranges_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            datatype, minimum, _null, maximum = line.split()
            ranges[datatype] = (float(minimum), float(maximum))
    rna_range = (np.log10(ranges['gene'][0] + 1), np.log10(ranges['gene'][1] + 1))
    cnv_range = (ranges['cnv'][0], ranges['cnv'][1])
    # 2 rows
    shape = (2, len(genes))
    # Instantiate the vector
    vector = np.zeros(shape, dtype=np.float32)
    cnv_intervals = patient.cnv_values()
    rna_genes = patient.gene_values()

    if cnv_intervals is None or rna_genes is None:
        return # Don't generate anything

    for gene in genes:
        # Fill in CNV data
        ensembl_entries = gene_intervals.get_gene_entries(gene)
        ensembl_entry = sorted(ensembl_entries, key=lambda x: int(x['gene_version']))[-1]
        start, stop = ensembl_entry['start'], ensembl_entry['stop']
        overlapping_cnv_intervals = cnv_intervals[ensembl_entry['chromosome']][start:stop]
        raw_cnv_value = np.median([x.data for x in overlapping_cnv_intervals])
        assert cnv_range[0] <= raw_cnv_value <= cnv_range[1]
        rel_cnv_value = (raw_cnv_value - cnv_range[0]) / (cnv_range[1] - cnv_range[0])
        vector[0][genes.index(gene)] = rel_cnv_value
        # Fill in RNA expression data
        raw_rna_value = np.log10(rna_genes[gene] + 1)
        assert rna_range[0] <= raw_rna_value <= rna_range[1]
        rel_rna_value = (raw_rna_value - rna_range[0]) / (rna_range[1] - rna_range[0])
        vector[1][genes.index(gene)] = rel_rna_value

    print(vector)
    print(vector.shape)
    if not os.path.isdir(TENSORS_DIR):
        os.makedirs(TENSORS_DIR)
    array_file_path = os.path.join(TENSORS_DIR, "%s.expression.array" % patient.barcode)
    with open(array_file_path, 'wb') as handle:
        pickle.dump(vector, handle)
    print("Generated %s in %f seconds" % (array_file_path, time.time() - start_time))


def create_gene_intersection_dimensional_vector(patient, gene_intervals, genes_file, ranges_file):
    """
    Creates a 3-dimensional image where the vertical axis is chromosome, horizontal is genes, and
    color channel is data type
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)
    assert isinstance(gene_intervals, gtf_parser.Gtf)

    start_time = time.time()

    # Load in the genes into a SortedDict<chromosome<SortedSet>>
    genes_sort_by_chromosome = sortedcontainers.SortedDict()
    with open(genes_file, 'r') as handle:
        for line in handle:
            gene = line.rstrip()
            matching_ensembl_entries = gene_intervals.get_gene_entries(gene)
            ensembl_entry = sorted(matching_ensembl_entries, key=lambda x: int(x['gene_version']))[-1]
            chromosome, start, stop = ensembl_entry['chromosome'], ensembl_entry['start'], ensembl_entry['stop']
            if chromosome not in genes_sort_by_chromosome:
                genes_sort_by_chromosome[chromosome] = sortedcontainers.SortedList(key=lambda x: (x['start'], x['stop']))
            genes_sort_by_chromosome[chromosome].add(ensembl_entry)
    # In total, there are 15368 genes
    # The largest per-chromosome set of genes has 1586 genes in it
    # print(sum([len(x) for x in genes_sort_by_chromosome.values()]))
    # print(max([len(x) for x in genes_sort_by_chromosome.values()]))

    # Stack/pack the genes such that each row has approximately the same number of genes, and such
    # that we minimize the amount of padding we need to do
    chromosomes_by_increasing_size = sortedcontainers.SortedList(
        [chrom for chrom in genes_sort_by_chromosome],
        key=lambda x: len(genes_sort_by_chromosome[x])
    )
    num_chromosomes = len(chromosomes_by_increasing_size)
    max_row_length = 1600  # Approximate size of the maximum row size that we want. This would get us about 10*16 under ideal packing
    all_rows, this_row, row_lengths, this_row_length = [], [], [], 0
    while chromosomes_by_increasing_size:  # While list still has elements in it
        # print(chromosomes_by_increasing_size)
        # print(list(itertools.chain.from_iterable(all_rows)) + this_row)
        biggest_chrom = chromosomes_by_increasing_size.pop(-1)
        big_size = len(genes_sort_by_chromosome[biggest_chrom])
        try:  # The case where we still have two or more elements left to pop out
            smallest_chrom = chromosomes_by_increasing_size.pop(0)
            small_size = len(genes_sort_by_chromosome[smallest_chrom])
            # Try in terms of ever decreasing size of extending the current list
            if big_size + small_size + this_row_length < max_row_length:
                this_row.append(biggest_chrom)
                this_row.append(smallest_chrom)
                this_row_length += big_size + small_size
            elif big_size + this_row_length < max_row_length:
                this_row.append(biggest_chrom)
                chromosomes_by_increasing_size.add(smallest_chrom)
                this_row_length += big_size
            elif small_size + this_row_length < max_row_length:
                this_row.append(smallest_chrom)
                chromosomes_by_increasing_size.add(biggest_chrom)
                this_row_length += small_size
            else:
                # We can no longer grow the current row - add it to all rows, and reset values
                # also reinsert the two chromosomes back in
                all_rows.append(this_row)
                row_lengths.append(this_row_length)
                this_row, this_row_length = [], 0  # Reset
                chromosomes_by_increasing_size.append(biggest_chrom)  # Reinsert
                chromosomes_by_increasing_size.insert(0, smallest_chrom)
        except IndexError:  # There is one chromosome left - either add it to this row or add it to the next row
            assert not chromosomes_by_increasing_size
            if big_size + this_row_length < max_row_length:
                this_row.append(chromosome)
                this_row_length += big_size
                all_rows.append(this_row)
                row_lengths.append(this_row_length)
            else:
                # Add it to a new row
                all_rows.append(this_row)
                row_lengths.append(this_row_length)
                all_rows.append([biggest_chrom])
                row_lengths.append(big_size)
    # If the current row being processed is not empty, append it
    if this_row:# and this_row_length > 0:
        all_rows.append(this_row)
        row_lengths.append(this_row_length)
    assert len(row_lengths) == len(all_rows)  # Make sure that our counters and our actual list of chromsomes stayed in sync
    assert sum([len(x) for x in all_rows]) == num_chromosomes  # Make sure we didn't miss any chromosomes in the packing process

    # Result of above looks like:
    # ['chr1']
    # ['chr19', 'chr21', 'chr18']
    # ['chr2', 'chr13', 'chr22']
    # ['chr11', 'chr20']
    # ['chr3', 'chr15']
    # ['chr17', 'chr14']
    # ['chr12', 'chr8']
    # ['chr6', 'chr16']
    # ['chr7', 'chrX']
    # ['chr5', 'chr4']
    # ['chr10', 'chr9']

    # Load in the ranges file
    ranges = {}
    with open(ranges_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            datatype, minimum, _null, maximum = line.split()
            ranges[datatype] = (float(minimum), float(maximum))
    rna_range = (np.log10(ranges['gene'][0] + 1), np.log10(ranges['gene'][1] + 1))
    cnv_range = (ranges['cnv'][0], ranges['cnv'][1])

    shape = (len(all_rows), max_row_length, 2)  # height, width, channels
    cnv_intervals = patient.cnv_values()
    rna_genes = patient.gene_values()
    if cnv_intervals is None or rna_genes is None:
        return # Don't generate anything
    # Instantiate the vector
    vector = np.zeros(shape, dtype=np.float32)
    for chromosome, genes in genes_sort_by_chromosome.items():
        row_index = [i for i, row in enumerate(all_rows) if chromosome in row][0]
        col_index = 0
        for ensembl_entry in genes:
            gene = ensembl_entry['gene_name']
            start, stop = ensembl_entry['start'], ensembl_entry['stop']
            overlapping_cnv_intervals = cnv_intervals[ensembl_entry['chromosome']][start:stop]
            raw_cnv_value = np.median([x.data for x in overlapping_cnv_intervals])
            assert cnv_range[0] <= raw_cnv_value <= cnv_range[1]
            rel_cnv_value = (raw_cnv_value - cnv_range[0]) / (cnv_range[1] - cnv_range[0])
            vector[row_index][col_index][0] = rel_cnv_value
            # Fill in RNA expression data
            raw_rna_value = np.log10(rna_genes[gene] + 1)
            assert rna_range[0] <= raw_rna_value <= rna_range[1]
            rel_rna_value = (raw_rna_value - rna_range[0]) / (rna_range[1] - rna_range[0])
            vector[row_index][col_index][1] = rel_rna_value
            col_index += 1
    # print(vector)
    print(vector.shape)
    if not os.path.isdir(TENSORS_DIR):
        os.makedirs(TENSORS_DIR)
    array_file_path = os.path.join(TENSORS_DIR, "%s.expression.array" % patient.barcode)
    with open(array_file_path, 'wb') as handle:
        pickle.dump(vector, handle)
    # The output of this is 16000 x 11        
    print("Generated %s in %f seconds" % (array_file_path, time.time() - start_time))

def create_full_union_single_vector(patient, gene_intervals, breakpoints_file, ranges_file):
    """
    Uses every single value in the union of all the datatypes, much like the above function.
    The difference is that in the above, we are compositing into a 2D image where the vertical
    axis is thecchromosome, and the horizontal axis is position along that chromosome. Here, all
    the chromosomes sit along the same 1D axis, in alphabetical order.

    For some reason, the png fiels produced by thsi fucntion can't be opened by windows photo viewer
    """
    assert isinstance(patient, tcga_parser.TcgaPatient)
    assert isinstance(gene_intervals, gtf_parser.Gtf)

    start_time = time.time()
    # reads in the data as both interval trees and in sorted dicts
    try:
        rna_intervals = gene_to_interval(patient.gene_values(), gene_intervals)
        rna_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in rna_intervals.items()}
        protein_intervals = gene_to_interval(patient.prot_values(), gene_intervals)
        protein_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in protein_intervals.items()}
        cnv_intervals = patient.cnv_values()
        cnv_intervals_sorted = {chromosome:interval_to_sorteddict(itree) for chromosome, itree in cnv_intervals.items()}
    except AttributeError:
        # Some data may be missing for some patients - for those we simply dont' generate an image
        return None

    # # Read in the ranges and breakpoints file
    breakpoints = {}
    with open(breakpoints_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            chromosome, points = line.split(": ")
            breakpoints[chromosome] = sortedcontainers.SortedList([int(x) for x in points.split(",")])
    ranges = {}
    with open(ranges_file, 'r') as handle:
        for line in handle:
            line = line.rstrip()
            datatype, minimum, _null, maximum = line.split()
            ranges[datatype] = (float(minimum), float(maximum))

    # In total we have 328934 breakpoints
    # https://stackoverflow.com/questions/12062920/how-do-i-create-an-image-in-pil-using-a-list-of-rgb-tuples
    width = sum([len(x) for x in breakpoints.values()])
    chr_cum_sum = {} # Describes how many breakpoints occured before the chromosome, noninclusive
    for chromosome in breakpoints.keys():
        chr_cum_sum[chromosome] = sum([len(bps) for chrom, bps in breakpoints.items() if chrom < chromosome])
    # Create the template for the image that we are going to create
    height = 1
    channels = 2 # RGB
    dimensions = (height, width, channels)
    print(dimensions)
    img = np.zeros(dimensions, dtype=np.float32) # unsigned 8-bit integers are 0-255
    # We walk through the chromosomes
    # for channel_index, sorted_intervals in enumerate([cnv_intervals_sorted, rna_intervals_sorted, protein_intervals_sorted]):
    for channel_index, sorted_intervals in enumerate([cnv_intervals_sorted, rna_intervals_sorted]):
        for chromosome in breakpoints.keys():
            try:
                values_for_chromosome = sorted_intervals[chromosome]
            except KeyError:
                # This chromosome doesn't exist for this datatype - oh well let's move on
                continue
            for start_stop_tuple, value in values_for_chromosome.items():
                # Normalize the value to be within the 0-255 range
                if channel_index == 0:
                    minimum, maximum = ranges['cnv']
                elif channel_index == 1:
                    minimum, maximum = ranges['gene']
                    # If it is gene data, convert it to a log scale to bring its order of magnitude
                    # to a level similar to that of cnv/prot data
                    minimum, maximum = np.log10(minimum + 1), np.log10(maximum + 1) # Convert to log
                    value = np.log10(value + 1)
                # elif channel_index == 2:
                #     minimum, maximum = ranges['prot']
                else:
                    raise ValueError("Unrecognized channel index: %i" % channel_index)
                if not value_within_range(value, minimum, maximum):
                    raise ValueError("%s WARNING: Given value %f does not fall in the min/max range: %f/%f" % (patient.barcode, value, minimum, maximum))
                value_normalized = np.float32((float(value) - float(minimum)) / float(maximum))

                start, stop = start_stop_tuple # Figure out the positions where it starts/stops
                start_index = breakpoints[chromosome].index(start) # Figure out where those coordinates lie on the list of sorted breakpoints
                stop_index = breakpoints[chromosome].index(stop)
                if not stop_index > start_index:
                    print("%s WARNING: Start index (%i) is not less than stop index (%i) for channel %i, chromosome %s" % (patient.barcode, start, stop, channel_index, chromosome))
                for col_index in range(start_index, stop_index + 1): # +1 to be inclusive of the stop index
                    col_index_with_offset = col_index + chr_cum_sum[chromosome]
                    assert col_index_with_offset < width
                    for i in range(height):
                        img[i][col_index_with_offset][channel_index] = value_normalized
    if not os.path.isdir(TENSORS_DIR):
        os.makedirs(TENSORS_DIR)
    array_file_path = os.path.join(TENSORS_DIR, "%s.expression.array" % patient.barcode)
    with open(array_file_path, 'wb') as handle:
        pickle.dump(img, handle)
    print("Generated %s in %f seconds" % (array_file_path, time.time() - start_time))

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

    breakpoints_file = os.path.join(tcga_analysis.RESULTS_DIR, "breakpoints.txt")
    ranges_file = os.path.join(tcga_analysis.RESULTS_DIR, "ranges.txt")
    patients = []
    for patient_file in tcga_patient_files:
        with open(patient_file, 'rb') as handle:
            patient = pickle.load(handle)
        assert isinstance(patient, tcga_parser.TcgaPatient)
        patients.append(patient)
    print("Finished loading in patient objects")
    #     create_image_full_union(
    #         patient,
    #         ensembl_genes,
    #         os.path.join(tcga_analysis.RESULTS_DIR, "breakpoints.txt"),
    #         os.path.join(tcga_analysis.RESULTS_DIR, "ranges.txt")
    #     )
    # Create the images in parallel
    pool = multiprocessing.Pool(6)
    # pool.map(functools.partial(create_image_full_union, gene_intervals=ensembl_genes, breakpoints_file=breakpoints_file, ranges_file=ranges_file), patients)
    # pool.map(functools.partial(create_image_full_union_single_vector, gene_intervals=ensembl_genes, breakpoints_file=breakpoints_file, ranges_file=ranges_file), patients)
    # pool.map(functools.partial(create_full_union_single_vector, gene_intervals=ensembl_genes, breakpoints_file=breakpoints_file, ranges_file=ranges_file), patients)
    # pool.map(functools.partial(create_gene_intersection_single_vector, gene_intervals=ensembl_genes, genes_file=tcga_analysis.COMMON_GENES_FILE, ranges_file = ranges_file), patients)
    pool.map(functools.partial(create_gene_intersection_dimensional_vector, gene_intervals=ensembl_genes,
                               genes_file=tcga_analysis.COMMON_GENES_FILE, ranges_file=ranges_file), patients)
    # for patient in patients:
    #     create_gene_intersection_single_vector(
    #         patient,
    #         ensembl_genes,
    #         tcga_analysis.COMMON_GENES_FILE,
    #         ranges_file
    #     )

if __name__ == "__main__":
    main()
