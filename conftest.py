from os import environ, makedirs

from pytest import fixture


def pytest_addoption(parser):
    parser.addoption(
        "--debug-graph",
        action="store_true",
        default=False,
        help="set debug=True for all the graphs in tests",
    )

    parser.addoption(
        "--output-path",
        default="output/tests",
        help="choose the location of output materials",
    )


@fixture(scope="session")
def output_path(request):
    loc = request.config.option.output_path
    makedirs(loc, exist_ok=True)
    return loc


@fixture(scope="session")
def debug_graph(request):
    return request.config.option.debug_graph


@fixture()
def test_name():
    """Returns corrected full name of a test."""
    name = environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    name = name.replace("[", "_").replace("]", "")
    return name
