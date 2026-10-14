import pytest


@pytest.fixture
def log_dir(tmp_path):
    return str(tmp_path / "logs")
