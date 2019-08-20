import pytest

from sme.large_file_storage import LargeFileStorage
from sme.config import Config


@pytest.fixture
def lfs_nlte():
    config = Config()
    server = config["data.file_server"]
    storage = config["data.nlte_grids"]
    cache = config["data.cache.nlte_grids"]
    pointers = config["data.pointers.nlte_grids"]
    lfs_nlte = LargeFileStorage(server, pointers, storage, cache)
    return lfs_nlte


def lfs_atmo():
    config = Config()
    server = config["data.file_server"]
    storage = config["data.atmospheres"]
    cache = config["data.cache.atmospheres"]
    pointers = config["data.pointers.atmospheres"]
    lfs_atmo = LargeFileStorage(server, pointers, storage, cache)
    return lfs_atmo


@pytest.fixture(name="lfs_atmo")
def lfs_atmo_fixture():
    return lfs_atmo()


def lfs_available():
    lfs = lfs_atmo()
    return lfs.server.isUp()


# skipif_lfs = pytest.mark.skipif(lfs_available(), reason="LFS not available")
skipif_lfs = pytest.mark.skip(reason="LFS not yet available")
