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
    for folder in [lfs_nlte.current, lfs_nlte.cache]:
        for filename in listdir(folder):
            file_path = join(folder, filename)
            try:
                remove(file_path)
            except IsADirectoryError as e:
                pass
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


@pytest.fixture
def lfs_atmo():
    lfs_atmo = setup_atmo()
    yield lfs_atmo

    for folder in [lfs_atmo.current, lfs_atmo.cache]:
        for filename in listdir(folder):
            file_path = join(folder, filename)
            try:
                remove(file_path)
            except IsADirectoryError as e:
                pass
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


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
