"""
Functions and classes for creating images that are expressive of
per-patient expression profiles

Notes to self:
   * Convert gene names into the intervaltree, much like how the cnvs
   are represented. This allows us to preserve the level of detail
   granted by CNV data, and combine it with the more general information
   given by rna/protein expression data
"""

import os
import pickle
import intervaltree
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


##### HERE ARE THE UTILITY FUNCTIONS FOR THEM #####


def gene_to_interval(gene_dict, gtf, verbose=True):
    """
    Converts a dictionary that maps genes to entries to a intervaltree of
    entries. Things that have no corresponding intervals associated in the
    given gtf are left out. The gtf should be read in for genes, since that
    is what we are querying
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
            gtf_entries = [gtf_entries[accept_indicies[0]]]
        itree[gtf_entries[0]['chromosome']] = gtf_entries[0]
        counter += 1
    if verbose:
        print("Converted %i/%i entries from gene dict to intervaltree" % (counter, len(gene_dict)))
    return itree


if __name__ == "__main__":
    # Load in a dummy file
    example = os.path.join(tcga_parser.DATA_ROOT, "tcga_patient_objects", "TCGA-UL-AAZ6.pickled")
    ensembl_genes = gtf_parser.Gtf(os.path.join(tcga_parser.DRIVE_ROOT, "Homo_sapiens.GRCh37.87.gtf"),
                                   "gene", set(['ensembl_havana']))
    with open(example, 'rb') as handle:
        patient = pickle.load(handle)
        assert isinstance(patient, tcga_parser.TcgaPatient)
        gene_to_interval(patient.gene_exp, ensembl_genes)
