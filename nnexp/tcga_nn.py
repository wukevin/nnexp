"""
The neural network itself!
"""
import tensorflow as tf
import sys
import os
import re
import glob
import pickle
import sortedcontainers
import numpy as np

import tcga_imaging
import tcga_parser

def build_one_hot_encoding(patients):
    """
    Builds a one-hot encoding table for the given list of patients
    """
    # Sanity check the input data
    for patient in patients:
        assert isinstance(patient, tcga_parser.TcgaPatient)

    # Define which keys we're interested in, and that they are present
    # across all the patients
    keys_of_interest = [
        # "er_status_by_ihc",
        "pr_status_by_ihc"
        # "her2_status_by_ihc"
    ]
    for key in keys_of_interest:
        for patient in patients:
            assert key in patient.clinical

    allowed_values = ["Positive", "Negative"]
    patients_filtered = [x for x in patients if x.clinical['pr_status_by_ihc'] in allowed_values]
    # Build the matrix
    onehot = tf.one_hot([allowed_values.index(x.clinical['pr_status_by_ihc']) for x in patients_filtered], 2)
    return onehot, patients_filtered


def convolution_neural_network():
    pass


def main():
    """
    """
    # Locate and load in the pickled patient records and the images
    patient_files = glob.glob(os.path.join(tcga_parser.DATA_ROOT, "tcga_patient_objects", "TCGA*.pickled"))
    assert len(patient_files) > 0
    patients = []
    for pickled_object in patient_files:
        with open(pickled_object, 'rb') as handle:
            patients.append(pickle.load(handle))
    images = glob.glob(os.path.join(tcga_imaging.IMAGES_DIR, "*.png"))
    barcodes_with_images = sortedcontainers.SortedSet([re.findall(tcga_parser.TCGA_BARCODE_REGEX, os.path.basename(x))[0] for x in images])
    # Filter the list of patients by those which we have images for and sort them
    # by alphebetical ordering of their barcodes
    patients = sortedcontainers.SortedList([x for x in patients if x.barcode in barcodes_with_images], key=lambda x: x.barcode)
    assert len(patients) > 0
    # Build the one-hot encoding table
    build_one_hot_encoding(patients)


if __name__ == "__main__":
    main()
