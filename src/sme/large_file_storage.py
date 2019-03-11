"""
System to store large data files on a server
Load them whem required by the user
Update the pointer file on github when new datafiles become available

Pro: Versioning is effectively done by Git
Con: Need to run server
"""

import os
import shutil
import hashlib
import logging


class LargeFileStorage:
    def __init__(self, server, pointers, storage, cache):
        #:Server: Large File Storage Server address
        self.server = Server(server)
        #:dict(fname:hash): points from a filename to the current newest object id, usually a hash
        self.pointers = pointers
        #:Directory: directory of the current data files
        self.current = Directory(storage)
        #:Directory: directory for the cache
        self.cache = Directory(cache)

    def hash(self, filename):
        """ hash a file """
        hasher = hashlib.sha256()
        blocksize = 65536
        with open(filename, "rb") as f:
            buf = f.read(blocksize)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(blocksize)
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
            current_hash = self.hash(self.current[key])
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
            pointers[name] = self.hash(self.current[name])

        self.pointers = pointers
        return pointers


class Directory:
    def __init__(self, path):
        self.path = os.path.expandvars(os.path.expanduser(path))

    def __contains__(self, key):
        return key in os.listdir(self.path)

    def __getitem__(self, key):
        path = os.path.join(self.path, key)
        if os.path.isdir(path):
            return Directory(path)
        else:
            return path

    def __setitem__(self, key, value):
        path = os.path.join(self.path, key)
        os.symlink(value, path)


class Server:
    def __init__(self, url):
        self.url = url

    def download(self, fname, location):
        # wget ?
        pass


if __name__ == "__main__":
    datafile = "debug.txt"
    server = "localhost"
    storage = os.path.expanduser("~/.sme/atmospheres")
    cache = os.path.expanduser("~/.sme/atmospheres/cache")

    pointers = None

    lfs = LargeFileStorage(server, pointers, storage, cache)
    pointers = lfs.generate_pointers()

    location = lfs.get(datafile)

    print(location)
