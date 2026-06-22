"""Configuration utilities.

Loads environment variables from `.env` and exposes API keys and configuration
constants as module-level attributes. All access to secrets must go through this
module — never hardcode or access os.environ directly elsewhere.

Raises:
    EnvironmentError: if any required environment variable is missing.
"""

import os
from dotenv import load_dotenv

# Load .env into os.environ if it exists
load_dotenv()


def _get_env(key: str, description: str) -> str:
    """Retrieve an environment variable with a clear error message if missing.

    Args:
        key: environment variable name
        description: human-readable description of where to obtain the value

    Returns:
        the environment variable value

    Raises:
        EnvironmentError: if the variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"{key} is not set. {description}"
        )
    return value


# Anthropic API key for Claude model access
ANTHROPIC_API_KEY = _get_env(
    "ANTHROPIC_API_KEY",
    "Get it from https://console.anthropic.com/account/keys",
)

# FRED API key for Federal Reserve Economic Data
FRED_API_KEY = _get_env(
    "FRED_API_KEY",
    "Get it from https://fredaccount.stlouisfed.org/apikeys",
)

# SEC API User-Agent string (required for EDGAR requests)
SEC_USER_AGENT = _get_env(
    "SEC_USER_AGENT",
    "Use a descriptive string like 'FinSight-Agent ({email})'",
)


__all__ = ["ANTHROPIC_API_KEY", "FRED_API_KEY", "SEC_USER_AGENT"]
