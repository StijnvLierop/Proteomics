import pytest
import pathlib

RESOURCE_DIR = pathlib.Path(__file__).parent.resolve() / 'resources'

@pytest.fixture
def pure_only_file():
    return RESOURCE_DIR / "pure_only_test.xlsx"

@pytest.fixture
def combi_only_file():
    return RESOURCE_DIR / "combi_only_test.xlsx"