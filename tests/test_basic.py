import mip2


def test_version_matches():
    # __version__ must be a non-empty string
    assert isinstance(mip2.__version__, str)
    assert mip2.__version__ != ""
