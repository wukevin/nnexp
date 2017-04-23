#!/usr/bin/env python3
"""
Functions for parsing TCGA data. Doesn't do any processing, but provides a
set of commands for easily parsing data
"""
import os
import sys
import glob
import re
import xml
import csv
import constants
import collections
import subprocess
import tarfile
import shutil

# Constants for where files are located
if sys.platform == 'win32' or sys.platform == "win64":
    DRIVE_ROOT = "E:\\"
elif sys.platform == "linux2" or sys.platform == "linux":
    DRIVE_ROOT = "/mnt/e/"
elif sys.platform == "darwin":
    DRIVE_ROOT = "/Volumes/Data"
else:
    raise RuntimeError("Unrecognized OS: %s" % sys.platform)
# Manifest
MANIFEST_FILE = os.path.join(DRIVE_ROOT, "gdc_manifest.geneexp_cnv_clinical_methylation_protexp.2017-04-08T23-26-44.308108.tsv")
# Contains all the downloaded data
DATA_ROOT = os.path.join(DRIVE_ROOT, "TCGA-BRCA")
# This file contains most of the clinical data and annotations that we'll be focusing on
CLINICAL_PATIENT_BRCA = os.path.join(DRIVE_ROOT, "TCGA-BRCA", "735bc5ff-86d1-421a-8693-6e6f92055563",
                                     "nationwidechildrens.org_clinical_patient_brca.txt")

TCGA_BARCODE_REGEX = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"


class TcgaPatient(object):
    """
    Object that stores all the information for a given TCGA case. Namely, we want
    to focus on storing the following, for now
    - Clinical data
    - CNV data
    - Gene expression data
    - Protein expression data
    """
    def __init__(self, tcga_id):
        """Initializes this object via its TCGA ID. The other attributes are left as empty placeholders"""
        # Check that the id matches our regex pattern
        matches = re.findall(TCGA_BARCODE_REGEX, tcga_id)
        if len(matches) != 1 or matches[0] != tcga_id:
            raise ValueError("Unrecognized TCGA ID: %s" % tcga_id)
        self.barcode = tcga_id
        self.clinical = {}
        self.cnv = {}
        self.gene_exp = {}
        self.prot_exp = {}

    def add_clinical_data(self, clinical):
        self.clinical = clinical

    def attach_relevant_cases(self, list_of_files):
        """Given a whole list of files, find relevant files to fill in the fields"""
        pass


class TcgaFileFinder(object):
    """
    """
    def __init__(self, data_root_dir, manifest_file):
        """
        If manifest is supplied, we will construct the index using that
        instead of walking through the directory itself
        """
        self.root = data_root_dir
        self.accepted_keys = ["barcode", "uuid"]

        # Build an index based on the manifest so we can quickly locate files
        self.filemap = {} # This will map the basename of each file to the full path to that file
        if not os.path.isfile(manifest_file):
            raise RuntimeError("Given manifest doesn't exist: %s" % manifest_file)
        with open(manifest_file) as handle:
            dictreader = csv.DictReader(handle, delimiter="\t")
            manifest_entries = [x for x in dictreader]
        for entry in manifest_entries:
            self.filemap[entry['filename']] = os.path.join(os.path.abspath(data_root_dir),
                                                           entry['id'], entry['filename'])

        # Build the mapping between UUID and barcodes
        clinical_patient_brca = self.filemap["nationwidechildrens.org_clinical_patient_brca.txt"]
        mappings = create_barcode_uuid_mapping(clinical_patient_brca)
        self.uuid_to_barcode, self.barcode_to_uuid = mappings

        # Get the unique SDRFs contained in this data root dir
        # These mage-tab tar.gz files contain SDRFs that allow mapping between filenames and IDs

        # Dictionary mapping unique mage-tab basenames to where all the copies of that mage-tab
        # basenames are
        sdrf_extracted_folder = os.path.join(self.root, "SDRFs")
        if os.path.isdir(sdrf_extracted_folder): # Remove the folder if it's there to make sure always up to date
            shutil.rmtree(sdrf_extracted_folder)
        os.makedirs(sdrf_extracted_folder)

        sdrf_directory = {} # Maps archive names to the sdrf files
        sdrf_dict = collections.defaultdict(list)
        magetab_pattern = os.path.join(data_root_dir, "*", "*mage-tab*.tar.gz")
        magetab_matches = glob.glob(magetab_pattern)
        for match in magetab_matches:
            sdrf_dict[os.path.basename(match)].append(match)
        for magetab_archive in sorted(sdrf_dict.keys()):
            # Gather md5 checksums for every alleged copy
            check_md5 = False # We've already checked once - no need to do it again, as it takes a long time
            if check_md5:
                md5_checksums = set([subprocess.check_output(["md5sum", x]).split()[0] for x in sdrf_dict[magetab_archive]])
                assert len(md5_checksums) == 1
            # Extract these to a separate location
            magetab_name_useful = self._get_magetab_name(magetab_archive)
            with tarfile.open(sdrf_dict[magetab_archive][0], "r:gz") as archive:
                archive.extractall(sdrf_extracted_folder)
            # Make sure that the extraction was successful, and that each contains a sdrf
            sdrf_pattern = os.path.join(sdrf_extracted_folder, magetab_name_useful, "*.sdrf.txt")
            sdrf_matches = glob.glob(sdrf_pattern)
            if len(sdrf_matches) != 1:
                raise RuntimeError("Cannot find sdrf under: %s" % os.path.join(sdrf_extracted_folder, magetab_name_useful))
            sdrf_file = sdrf_matches[0]
            sdrf_directory[magetab_archive] = sdrf_file
        
        # Read in the SDRFs
        self.sdrfs = {}
        for archivename, sdrffile in sdrf_directory.items():
            self.sdrfs[archivename] = read_sdrf(sdrffile)

    def _get_magetab_name(self, magetab_basename):
        """Given the magetab basename, get substring omitting mage-tab and version number"""
        # pattern = r"\.mage-tab\.[0-9]*\.[0-9]*\.[0-9]*\.tar\.gz"
        pattern = r"\.tar\.gz"
        return re.sub(pattern, "", os.path.basename(magetab_basename))

    def get_cnv_files(self):
        """
        Returns a dictionary mapping barcode to dictionary of cnv files. Only returns cnv files assoc with hg19
        Dictionary type is dict<BARCODE, dict<type, cnv_filename>>
        """
        cnv_sdrf_archive = "broad.mit.edu_BRCA.Genome_Wide_SNP_6.mage-tab.1.2024.0.tar.gz"
        # if key not in self.accepted_keys:
        #     raise ValueError("%s must be one of: %s" % (key, " ".join(self.accepted_keys)))
        retval = collections.defaultdict(dict)
        for basename, fullname in self.filemap.items():
            if re.search(r"_hg19.seg.txt$", basename) is not None: # We found a CNV hg19 file
                # This is the directory that we are currently in
                curr_dir = os.path.dirname(fullname)
                # archive for CNVs is broad.mit.edu_BRCA.Genome_Wide_SNP_6.mage-tab.1.2024.0.tar.gz
                assert os.path.isfile(os.path.join(curr_dir, cnv_sdrf_archive))
                archive_key = cnv_sdrf_archive
                sdrf_table = self.sdrfs[archive_key]
                # Since teh table is a dictioanry of barcode --> items, iterate through those pairs
                for barcode, entries in sdrf_table.items():
                    for entry in entries: # For each row belonging to that barcode
                        for item in entry.values(): # For each item in that row
                            if item == basename: # If the item matches...
                                detailed_barcode = entry["Comment [TCGA Barcode]"]
                                detailed_barcode_tokenized = detailed_barcode.split("-")
                                sample_site = int(detailed_barcode_tokenized[3][:-1])
                                if sample_site < 10:
                                    sample_type = "tumor"
                                elif sample_site < 20:
                                    sample_type = "normal"
                                elif sample_site < 30:
                                    sample_type = "control"
                                else:
                                    raise ValueError("%i is not a recognized sample type" % sample_site)
                                retval[barcode][sample_type] = fullname
        # for barcode, filenames in retval.items():
        #     print(barcode)
        #     for sample_type, sample_filename in filenames.items():
        #         print("%s:\t%s" % (sample_type, sample_filename))
        return retval

    def get_rnaseq_files(self):
        """
        Returns a dictionary mapping barcode to a dictionary of RNASeqV2 files
        """
        # We want RSEM_genes_normalized
        rnaseq_sdrf_archives = [
            "unc.edu_BRCA.IlluminaHiSeq_RNASeqV2.mage-tab.1.12.0.tar.gz",
            "unc.edu_BRCA.IlluminaHiSeq_TotalRNASeqV2.mage-tab.1.1.0.tar.gz",
        ]
        retval = collections.defaultdict(dict)
        for basename, fullname in self.filemap.items():
            if re.search(r"rsem.genes.normalized_results$", basename) is not None:
                curr_dir = os.path.dirname(fullname)
                if os.path.isfile(os.path.join(curr_dir, rnaseq_sdrf_archives[0])):
                    archive_key = rnaseq_sdrf_archives[0]
                elif os.path.isfile(os.path.join(curr_dir, rnaseq_sdrf_archives[1])):
                    archive_key = rnaseq_sdrf_archives[1]
                else:
                    raise RuntimeError("Cannot find a valid archive filekey in %s" % curr_dir)
                sdrf_table = self.sdrfs[archive_key]
                for barcode, entries in sdrf_table.items():
                    for entry in entries:
                        for item in entry.values():
                            if item == basename:
                                detailed_barcode = entry["Comment [TCGA Barcode]"]
                                detailed_barcode_tokenized = detailed_barcode.split("-")
                                sample_site = int(detailed_barcode_tokenized[3][:-1])
                                if sample_site < 10:
                                    sample_type = "tumor"
                                elif sample_site < 20:
                                    sample_type = "normal"
                                elif sample_site < 30:
                                    sample_type = "control"
                                else:
                                    raise ValueError("%i is not a recognized sample type" % sample_site)
                                retval[barcode][sample_type] = fullname
        print(len(retval))
        for barcode, filenames in retval.items():
            print(barcode)
            for sampletype, samplefile in filenames.items():
                print("%s:\t%s" % (sampletype, samplefile))
        return retval
    def get_protexp_files(self):
        """
        Returns a dictionary mapping barcode to a dictionary of protein expression files
        """

def create_barcode_uuid_mapping(biotab_file):
    """Return a pair of dict mappings from uuid -> barcode and barcode -> uuid"""
    # Read in the lines of the file
    with open(biotab_file, 'r') as handle:
        lines = handle.readlines()
    header_line = lines[0]
    header_tokenized = header_line.split()
    uuid_column = header_tokenized.index("bcr_patient_uuid")
    barcode_column = header_tokenized.index("bcr_patient_barcode")

    # Walk through the content lines and extract the barcode mapping
    content_lines = lines[3:]
    uuid_to_barcode = {}
    barcode_to_uuid = {}
    for line in content_lines:
        line_tokenized = line.split()
        uuid = line_tokenized[uuid_column]
        barcode = line_tokenized[barcode_column]
        if uuid in uuid_to_barcode:
            raise ValueError("Found duplicate UUID: %s" % uuid)
        if barcode in barcode_to_uuid:
            raise ValueError("Found duplicate barcode: %s" % barcode)
        barcode_to_uuid[barcode] = uuid
        uuid_to_barcode[uuid] = barcode_column
    # Return the two mppings
    return uuid_to_barcode, barcode_to_uuid


def read_sdrf(sdrf_file):
    """Reads the given sdrf_file. Returns a dictionary <TCGA_BARCODE, DICT_OF_ELEMENTS>"""
    assert os.path.isfile(sdrf_file) and "sdrf" in os.path.basename(sdrf_file)
    with open(sdrf_file, 'r') as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        entries = [x for x in reader]
    retval = collections.defaultdict(list)
    for entry in entries:
        try:
            possible_barcodes = re.findall(TCGA_BARCODE_REGEX, entry["Comment [TCGA Barcode]"])
        except KeyError:
            # This should only be triggered by the RPPA SDRF
            assert "RPPA" in sdrf_file
            possible_barcodes = re.findall(TCGA_BARCODE_REGEX, entry["Extract Name"])
        if len(possible_barcodes) != 1:
            # We know that in the Agilent SDRF, there are lines that are just ->
            # We skip those fow now
            if not ("Agilent" in sdrf_file and entry["Comment [TCGA Barcode]"] == "->"):
                raise RuntimeError("%s: Cannot parse barcode from: %s. Skipping" % (sdrf_file, entry["Comment [TCGA Barcode]"]))
            continue
        barcode = possible_barcodes[0]
        retval[barcode].append(entry)
    return retval


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


def get_cnv_files(directory, uuid_to_barcode_map):
    """Returns a dictionary of TCGA IDs and their associated CNV files"""
    # We only want hg19
    cnv_pattern = os.path.join(os.path.abspath(directory), "*", "*_hg19.seg.txt")
    print(cnv_pattern)
    cnv_files = glob.glob(cnv_pattern)
    print(len(cnv_files))

def read_clinical_xml(clinical_xml_file):
    """
    Reads TCGA xml file
    """
    assert isinstance(clinical_xml_file, str)
    assert os.path.isfile(clinical_xml_file)
    with open(clinical_xml_file, 'r') as read_handle:
        pass


def read_biotab(biotab_file, replace_not_available=True, replace_not_applicable=True):
    """
    Reads TCGA Biotab file and returns it as a list of dictionaries.
    Additionally, replaces all [Not Available] fields with None for
    easier parsing later
    """
    assert isinstance(biotab_file, str)
    retval = []
    with open(biotab_file, 'r') as filehandle:
        lines = filehandle.readlines()
    # Remove the 2nd and 3rd lines of the file which are just additional metadata and do not
    # contain useful information
    lines.pop(1)
    lines.pop(1)
    parser = csv.DictReader(lines, delimiter="\t")
    for entry in parser:
        # Replace values in the dictionary that are not available/applicable for easy
        # parsing later on
        for key, value in entry.items():
            if value == "[Not Available]" and replace_not_available:
                entry[key] = None
            elif value == "[Not Applicable]" and replace_not_applicable:
                entry[key] = None
        assert 'bcr_patient_barcode' in entry # Make sure the TCGA barcode is there
        retval.append(entry)
    return retval


if __name__ == "__main__":
    # DATA_ROOT_DIR = os.getcwd()
    # clinical_xmls = get_samples_with_clinical_xml(DATA_ROOT_DIR)
    # print "Found %i clinical records" % len(clinical_xmls)

    # clinical_biotab = get_biotab_files(DATA_ROOT_DIR)
    # desired_biotabs = [x for x in clinical_biotab if "clinical_patient_brca" in x]
    # assert len(desired_biotabs) == 1
    # clinical_data = read_biotab(CLINICAL_PATIENT_BRCA)
    # for entry in clinical_data:
    pass
