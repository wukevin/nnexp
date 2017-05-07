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
import tcga_parser
from PIL import Image

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


if __name__ == "__main__":
    # Load in a dummy file
    example = os.path.join(tcga_parser.DATA_ROOT, "tcga_patient_objects", "TCGA-UL-AAZ6.pickled")
    with open(example, 'rb') as handle:
        patient = pickle.load(handle)
        create_image_full_gene_intersection(patient)
