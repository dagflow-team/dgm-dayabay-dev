from os import environ, makedirs

from pytest import fixture


def pytest_sessionstart(session):
    """Called after the Session object has been created and before performing
    collection and entering the run test loop.

    Create `tests/output` dir
    """
    makedirs("tests/output", exist_ok=True)
def pytest_addoption(parser):
    parser.addoption(
        "--debug-graph",
        action="store_true",
        default=False,
        help="set debug=True for all the graphs in tests",
    )


@fixture(scope="session")
def debug_graph(request):
    return request.config.option.debug_graph


@fixture()
def test_name():
    """Returns corrected full name of a test."""
    name = environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    name = name.replace("[", "_").replace("]", "")
    return name
