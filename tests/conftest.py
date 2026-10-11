import pytest

from relcomplete.gen import build


@pytest.fixture(scope="session")
def bench():
    return build(0)
