#!/usr/bin/env python3
"""
Processes TCGA data to produces the raw "images" that we
feed into the neural net
"""
import tcga_parser

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

    # cnv_files = filefinder.get_cnv_files()

if __name__ == "__main__":
    main()
