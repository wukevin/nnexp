#!/usr/bin/env python3
"""
Processes TCGA data to produces the raw TCGA patient objects. We
then feed these objects into downstream functions to generate
images based on the raw data, which are then fed into the neural net
"""
import tcga_parser
import gtf_parser
import glob
import time
import os
import pickle

def create_tcga_objects(biotab):
    """
    Creates the TCGA objects and filles in clincial data. The sole
    input argument, biotab, is meant to be the biotab file itself,
    not the contents
    """
    # dictionary mapping tcga ids to their corresponding object
    tcga_objects = {}
    clinical_data = tcga_parser.read_biotab(biotab)
    for entry in clinical_data:
        tcga_id = entry['bcr_patient_barcode']
        if tcga_id in tcga_objects:
            raise RuntimeError("Found duplicate TCGA ID: %s" % tcga_id)
        tcga_objects[tcga_id] = tcga_parser.TcgaPatient(tcga_id)
        tcga_objects[tcga_id].add_clinical_data(entry)
    return tcga_objects


def load_tcga_objects(root=tcga_parser.DATA_ROOT):
    """
    Load in the TCGA objects from the default direcrtory
    """
    pattern = os.path.join(
        root,
        "tcga_patient_objects",
        "TCGA*.pickled"
    )
    tcga_patient_files = glob.glob(pattern)
    if len(tcga_patient_files) == 0:
        raise RuntimeError("Found no files matching pattern:\n%s" % pattern)

    # Load in all the patients
    patients = []
    for patient_file in tcga_patient_files:
        with open(patient_file, 'rb') as handle:
            patient = pickle.load(handle)
        assert isinstance(patient, tcga_parser.TcgaPatient)
        patients.append(patient)

    return patients

def main():
    """"""
    # Create the objects, filling in clinical data
    tcga_objects = create_tcga_objects(tcga_parser.CLINICAL_PATIENT_BRCA)

    # Create dictionaries that represent the mappings between barcode and uuid
    # uuid_to_barcode, barcode_to_uuid = tcga_parser.create_barcode_uuid_mapping(tcga_parser.CLINICAL_PATIENT_BRCA)

    # Create the indexed file finder
    filefinder = tcga_parser.TcgaFileFinder(
        tcga_parser.DATA_ROOT,
        tcga_parser.MANIFEST_FILE
    )

    # Gather all the files
    start_time = time.time()
    cnv_files = filefinder.get_cnv_files()
    assert len(cnv_files) > 0
    rnaseq_files = filefinder.get_rnaseq_files()
    assert len(rnaseq_files) > 0
    protexp_files = filefinder.get_protexp_files()
    assert len(protexp_files) > 0
    print("Finished gathering files in %f seconds" % (time.time() - start_time))

    # Get the barcodes that have clinical, cnv, rnaseq, and protexp data
    common_barcodes = set([x for x in tcga_objects.keys() if x in cnv_files and x in rnaseq_files and x in protexp_files])

    # Filter out TCGA patients that don't have "complete" data
    tcga_objects = [x for x in tcga_objects.values() if x.barcode in common_barcodes]

    # Fill in the TCGA objects
    for tcga_case in tcga_objects:
        tcga_case.data_files['cnv'] = cnv_files[tcga_case.barcode]
        tcga_case.data_files['rnaseq'] = rnaseq_files[tcga_case.barcode]
        tcga_case.data_files['rppa'] = protexp_files[tcga_case.barcode]

    # Read in the files, then write these TCGA objects to disk
    tcga_objects_directory = os.path.join(
        tcga_parser.DATA_ROOT, "tcga_patient_objects"
    )
    if not os.path.isdir(tcga_objects_directory):
        os.makedirs(tcga_objects_directory)
    for obj in tcga_objects:
        # Read in the associated files, attaching them to the object
        obj.parse_attached_files()
        # Write to disk
        filename_output = os.path.join(tcga_objects_directory, "%s.pickled" % obj.barcode)
        with open(filename_output, 'wb') as handle:
            pickle.dump(obj, handle)

if __name__ == "__main__":
    main()
