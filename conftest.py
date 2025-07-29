from os import chdir, environ, getcwd, listdir, mkdir
from os.path import isdir

from pytest import fixture


def pytest_sessionstart(session):
    """Called after the Session object has been created and before performing
    collection and entering the run test loop.

    Create `output/` dir
    """
    if not isdir("output"):
        mkdir("output")


@fixture()
def testname():
    """Returns corrected full name of a test."""
    name = environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    name = name.replace("[", "_").replace("]", "")
    return name
