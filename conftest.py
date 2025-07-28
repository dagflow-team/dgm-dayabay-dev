from os import chdir, environ, getcwd, listdir, mkdir
from os.path import isdir

from pytest import fixture


def pytest_sessionstart(session):
    """Called after the Session object has been created and before performing
    collection and entering the run test loop.

    Automatic change path to the `dgm-dayabay-dev/tests` and create
    `tests/output` dir
    """
    while path := getcwd():
        if (lastdir := path.split("/")[-1]) == "tests":
            break
        elif ".git" in listdir(path):
            chdir("./tests")
            break
        else:
            chdir("..")
    if not isdir("output"):
        mkdir("output")


@fixture()
def testname():
    """Returns corrected full name of a test."""
    name = environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    name = name.replace("[", "_").replace("]", "")
    return name
