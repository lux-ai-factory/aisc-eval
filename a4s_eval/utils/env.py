"""Environment configuration module for A4S Evaluation.

This module defines environment variables and their default values used throughout
the A4S evaluation system. These can be overridden by setting actual environment
variables.
"""

import os

# Base URL for the API service, defaults to localhost:8000 if not specified
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Directory for caching downloaded models and datasets
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/cache")