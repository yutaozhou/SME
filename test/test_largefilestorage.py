import pytest

from os import listdir, makedirs, stat
from os.path import dirname, islink, exists
from shutil import rmtree

from pysme.large_file_storage import LargeFileStorage, Server
from pysme.config import Config


def lfs_available():
    config = Config()
    server = Server(config["data.file_server"])
    return server.isUp()


skipif_lfs = pytest.mark.skipif(lfs_available(), reason="LFS not available")


@pytest.fixture
def lfs_nlte():
    config = Config()
    server = config["data.file_server"]
    pointers = config["data.pointers.nlte_grids"]
    storage = "./lfs_test"
    cache = "./lfs_test/cache"
    makedirs(cache, exist_ok=True)
    lfs_nlte = LargeFileStorage(server, pointers, storage, cache)
    yield lfs_nlte
    rmtree(storage, ignore_errors=True)


@pytest.fixture
def lfs_atmo():
    config = Config()
    server = config["data.file_server"]
    pointers = config["data.pointers.atmospheres"]
    storage = "./lfs_test"
    cache = "./lfs_test/cache"
    makedirs(cache, exist_ok=True)
    lfs_atmo = LargeFileStorage(server, pointers, storage, cache)
    yield lfs_atmo
    rmtree(storage, ignore_errors=True)


@skipif_lfs
def test_load_atmo(lfs_atmo):
    fname = "marcs2012p_t0.0.sav"
    dname = lfs_atmo.get(fname)
    assert fname in listdir(dirname(dname))
    assert exists(dname)
