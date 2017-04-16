"""
Functions for parsing TCGA data
"""
import os
import sys
import glob
import re
import xml
import csv
import constants


def get_samples_with_clinical_xml(directory):
    """
    Return a dictionary of TCGA IDs and their associated files
    """
    retval = {}
    pattern = os.path.join(directory, "*", "*.xml")
    for xml_file in glob.glob(pattern):
        matches = re.findall(constants.TCGA_ID_PATTERN, xml_file)
        if len(matches) != 1:
            raise RuntimeError("Cannot find TCGA ID in %s" % xml_file)
        tcga_id = matches[0]
        if tcga_id in retval:
            raise RuntimeError("%s has already been seen" % tcga_id)
        retval[tcga_id] = os.path.abspath(xml_file)
    return retval


def get_biotab_files(directory):
    """
    Returns a list of biotab files
    """
    pattern = os.path.join(directory, "*", "nationwide*.txt")
    results = glob.glob(pattern)
    if len(results) == 0:
        raise RuntimeError("Could not find any biotab files")
    return results


def get_samples_with_cnv_data(directory):
    """
    Returns a dictionary of TCGA IDs and their associated CNV files
    """

def read_clinical_xml(clinical_xml_file):
    """
    Reads TCGA xml file
    """
    assert isinstance(clinical_xml_file, basestring)
    assert os.path.isfile(clinical_xml_file)
    with open(clinical_xml_file, 'r') as read_handle:
        pass


def read_biotab(biotab_file, replace_not_available=True):
    """
    Reads TCGA Biotab file and returns it as a list of dictionaries.
    Additionally, replaces all [Not Available] fields with None for
    easier parsing later
    """
    assert isinstance(biotab_file, basestring)
    retval = []
    with open(biotab_file, 'r') as filehandle:
        parser = csv.DictReader(filehandle, delimiter="\t")
        for entry in parser:
            for key, value in entry.iteritems():
                if value == "[Not Available]" and replace_not_available:
                    entry[key] = None
            assert 'bcr_patient_barcode' in entry # Make sure the TCGA barcode is there
            retval.append(entry)
    return retval


if __name__ == "__main__":
    DATA_ROOT_DIR = os.getcwd()
    clinical_xmls = get_samples_with_clinical_xml(DATA_ROOT_DIR)
    print "Found %i clinical records" % len(clinical_xmls)

    clinical_biotab = get_biotab_files(DATA_ROOT_DIR)
    desired_biotabs = [x for x in clinical_biotab if "clinical_patient_brca" in x]
    assert len(desired_biotabs) == 1
    clinical_data = read_biotab(desired_biotabs[0])
