import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def enable_fake_models():
    os.environ["ENABLE_FAKE_MODELS"] = "true"
