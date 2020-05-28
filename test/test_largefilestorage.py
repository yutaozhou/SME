import pytest

from os import listdir, makedirs, stat, remove
from os.path import dirname, islink, exists, join
from shutil import rmtree

from pysme.large_file_storage import LargeFileStorage, Server, setup_atmo, setup_nlte
from pysme.config import Config


def lfs_available():
    config = Config()
    server = Server(config["data.file_server"])
    return server.isUp()


skipif_lfs = pytest.mark.skipif(lfs_available(), reason="LFS not available")


@pytest.fixture
def lfs_nlte():
    lfs_nlte = setup_nlte()
    yield lfs_nlte


@pytest.fixture
def lfs_atmo():
    lfs_atmo = setup_atmo()
    yield lfs_atmo


@skipif_lfs
def test_load_atmo(lfs_atmo):
    fname = "marcs2012p_t0.0.sav"
    dname = lfs_atmo.get(fname)
    assert fname in listdir(dirname(dname))
    assert exists(dname)


@skipif_lfs
def test_load_nlte(lfs_nlte):
    fname = "marcs2012_Na.grd"
    dname = lfs_nlte.get(fname)
    assert fname in listdir(dirname(dname))
    assert exists(dname)
