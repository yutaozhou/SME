"""
System to store large data files on a server
Load them whem required by the user
Update the pointer file on github when new datafiles become available

Pro: Versioning is effectively done by Git
Con: Need to run server
"""

import hashlib
import logging
import os
import shutil

import wget
from ruamel.yaml import YAML


class LargeFileStorage:
    """
    Download large data files from data server when needed
    New versions of the datafiles are indicated in a 'pointer' file
    that includes the hash of the newest version of the files

    Raises
    ------
    FileNotFoundError
        If the datafiles can't be located anywhere
    """

    def __init__(self, server, pointers, storage, cache):
        #:Server: Large File Storage Server address
        self.server = Server(server)

        if isinstance(pointers, str):
            pointers = os.path.join(os.path.dirname(__file__), pointers)
            if not os.path.exists(pointers):
                with open(pointers, "w") as f:
                    f.writelines("")
            yaml = YAML(typ="safe")
            with open(pointers, "r") as f:
                pointers = yaml.load(f)
        #:dict(fname:hash): points from a filename to the current newest object id, usually a hash
        self.pointers = pointers
        #:Directory: directory of the current data files
        self.current = Directory(storage)
        #:Directory: directory for the cache
        self.cache = Directory(cache)
        #:dict(fname:hash): hashes of existing files, to avoid recalculation
        self._hashes = {}

    def hash(self, filename):
        """ hash a file """
        hasher = hashlib.sha3_512()
        blocksize = 8192  # 512 * 16
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blocksize), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get(self, key):
        """ make sure to get the latest version of the file """
        # Step 1: Check if the file is tracked and/or exists in the storage directory
        if key not in self.pointers:
            if key not in self.current:
                raise FileNotFoundError(
                    "File does not exist and is not tracked by the Large File system"
                )
            else:
                logging.warning(
                    "Data file exists, but is not tracked by the large file storage"
                )
                return key

        # Step 2: Check Pointer version, i.e. newest version
        newest = self.pointers[key]

        if key in self.current:
            # Step 3: If newest version == storage version, we are all good and can use it
            if key in self._hashes.keys():
                current_hash = self._hashes[key]
            else:
                current_hash = self.hash(self.current[key])
                self._hashes[key] = current_hash
            if current_hash == newest:
                return self.current[key]

        # Step 4: Otherwise check the cache for the requested version
        if newest in self.cache:
            logging.debug("Using cached version of datafile")
            self.current[key] = self.cache[newest]
            return self.current[key]

        # Step 5: If not in the cache, download from the server
        logging.info("Downloading newest version of the datafile from server")
        self.server.download(newest, self.cache)
        self.current[key] = self.cache[newest]
        return key

    def clean_cache(self):
        """ Remove unused cache files (from old versions) """
        used_files = self.pointers.values()
        for f in os.listdir(self.cache.path):
            if f not in used_files:
                os.remove(f)

    def generate_pointers(self):
        """ Generate the pointers dictionary from the existing storage directory """
        pointers = {}
        for name in os.listdir(self.current.path):
            if os.path.isfile(os.path.join(self.current.path, name)):
                pointers[name] = self.hash(self.current[name])

        self.pointers = pointers
        return pointers

    def move_to_cache(self):
        """ Move currently used files into cache directory and use symlinks insteadm, just if downloaded from a server """
        for name in self.current:
            fullpath = self.current[name]
            if not isinstance(fullpath, Directory) and not os.path.islink(fullpath):
                # Copy file
                shutil.copy(fullpath, self.cache[self.pointers[name]])
                os.remove(fullpath)
                self.current[name] = self.cache[self.pointers[name]]

    def create_pointer_file(self, filename):
        """ Create/Update the pointer file with new hashes """
        if self.pointers is None:
            raise RuntimeError("Needs pointers")

        yaml = YAML(typ="safe")
        yaml.default_flow_style = False

        with open(filename, "w") as f:
            yaml.dump(self.pointers, f)


class Directory:
    def __init__(self, path):
        self.path = os.path.expandvars(os.path.expanduser(path))

    def __contains__(self, key):
        return key in os.listdir(self.path)

    def __iter__(self):
        return iter(os.listdir(self.path))

    def __getitem__(self, key):
        path = os.path.join(self.path, key)
        if os.path.isdir(path):
            return Directory(path)
        else:
            return path

    def __setitem__(self, key, value):
        path = os.path.join(self.path, key)
        os.symlink(value, path)

    def __str__(self):
        return self.path


class Server:
    def __init__(self, url):
        self.url = url

    def download(self, fname, location):
        url = os.path.join(self.url, fname)
        wget.download(url, out=str(location))


if __name__ == "__main__":
    import config

    conf = config.Config()

    datafile = "atlas12.sav"
    server = conf["data.file_server"]
    storage = conf["data.atmospheres"]
    cache = conf["data.cache.atmospheres"]
    pointers = conf["data.pointers.atmopsheres"]

    lfs = LargeFileStorage(server, pointers, storage, cache)

    lfs.generate_pointers()
    lfs.move_to_cache()
    lfs.create_pointer_file(pointers)

    location = lfs.get(datafile)

    print(location)

    server = conf["data.file_server"]
    storage = conf["data.nlte_grids"]
    cache = conf["data.cache.nlte_grids"]
    pointers = conf["data.pointers.nlte_grids"]

    lfs = LargeFileStorage(server, pointers, storage, cache)

    lfs.generate_pointers()
    lfs.move_to_cache()
    lfs.create_pointer_file(pointers)
