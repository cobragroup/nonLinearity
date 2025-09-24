import mienc


def test_version_matches():
    # __version__ must be a non-empty string
    assert isinstance(mienc.__version__, str)
    assert mienc.__version__ != ""
