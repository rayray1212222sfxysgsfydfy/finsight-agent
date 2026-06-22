import importlib
import sys

import pytest


def _reload_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-example")
    monkeypatch.setenv("FRED_API_KEY", "fred-example")
    monkeypatch.setenv("SEC_USER_AGENT", "FinSight-Agent (test@example.com)")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    sys.modules.pop("src.utils.config", None)
    import src.utils.config as config
    return config


def test_config_constants_are_non_empty_strings(monkeypatch):
    config = _reload_config(monkeypatch)

    assert isinstance(config.ANTHROPIC_API_KEY, str)
    assert config.ANTHROPIC_API_KEY != ""
    assert isinstance(config.FRED_API_KEY, str)
    assert config.FRED_API_KEY != ""
    assert isinstance(config.SEC_USER_AGENT, str)
    assert config.SEC_USER_AGENT != ""


def test_anthropic_api_key_starts_with_sk_ant(monkeypatch):
    config = _reload_config(monkeypatch)

    assert config.ANTHROPIC_API_KEY.startswith("sk-ant-")


@pytest.mark.parametrize(
    "missing_key, keep_keys",
    [
        ("ANTHROPIC_API_KEY", ["FRED_API_KEY", "SEC_USER_AGENT"]),
        ("FRED_API_KEY", ["ANTHROPIC_API_KEY", "SEC_USER_AGENT"]),
        ("SEC_USER_AGENT", ["ANTHROPIC_API_KEY", "FRED_API_KEY"]),
    ],
)
def test_missing_required_env_raises_environment_error(monkeypatch, missing_key, keep_keys):
    for key in ["ANTHROPIC_API_KEY", "FRED_API_KEY", "SEC_USER_AGENT"]:
        if key == missing_key:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, f"{key.lower()}-example")

    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    sys.modules.pop("src.utils.config", None)

    with pytest.raises(EnvironmentError, match=missing_key):
        import src.utils.config as config  # noqa: F401
