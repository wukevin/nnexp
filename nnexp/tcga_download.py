import os
import sys
import csv
import argparse
import multiprocessing
import subprocess



class TcgaDownloader(multiprocessing.Process):
    """Multithreaded TCGA downloader"""
    def __init__(self, manifest_entry_queue, basepath, download=False):
        super(TcgaDownloader, self).__init__()
        self.input_queue = manifest_entry_queue
        self.download = download
        self.basepath = basepath

    def check_file(self, filepath, md5sum=None):
        """Checks file, returns true if exists and md5sum matches (if provided)"""
        if not os.path.isfile(filepath):
            return False
        if md5sum is not None:
            checksum = subprocess.check_output(['md5sum', filepath]).split()[0]
            if checksum != md5sum:
                return False
        return True

    def download_file(self):
        """Downloads the file"""
        raise NotImplementedError

    def run(self):
        """Override default run method"""
        while True:
            next_item = self.input_queue.get()
            if next_item is None:
                break
            # Process the dictionary item
            # Check if the file is there
            path_to_check = os.path.join(self.basepath, next_item["id"], next_item["filename"])
            if self.check_file(path_to_check, next_item['md5']):
                # print "Found: %s" % path_to_check
                continue

            # File is not found - download it
            print "NOT found: %s" % path_to_check


def parse_manifest(manifest):
    """
    Parses the TCGA manifest file and returns dictionary
    """
    with open(manifest, 'r') as handle:
        dictreader = csv.DictReader(handle, delimiter="\t")
        retval = [item for item in dictreader]
    return retval


def build_parser():
    """Builds the argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--directory", required=False, type=str, default=os.getcwd())
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--check")
    return parser


def main():
    """Runs the downloader"""
    parser = build_parser()
    args = parser.parse_args()

    manifest_entries = parse_manifest(args.manifest)
    # Fill the queue with dictionary entries
    queue = multiprocessing.Queue()
    for item in manifest_entries:
        queue.put(item)
    # Signal the end of the queue
    for i in range(args.threads):
        queue.put(None)

    # Start the queue processors
    downloaders = []
    for i in range(args.threads):
        process = TcgaDownloader(queue, args.directory)
        process.start()
        downloaders.append(process)
    # Finish the threads
    for downloader in downloaders:
        downloader.join()

if __name__ == "__main__":
    main()


