# The file is needed in order to setup the CWD
from os import mkdir
from os.path import isdir

from pytest import fixture


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.

    Automatic change path to the `tests` and create `tests/output` dir
    """
    if not isdir("output"):
        mkdir("output")


def pytest_addoption(parser):
    parser.addoption("--debug-graph", action="store_true", default=False)


@fixture(scope="session")
def debug_graph(request):
    return request.config.option.debug_graph
