import os
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="session")
def env_vars():
    """Ensure environment variables are loaded."""
    load_dotenv()
    return os.environ
