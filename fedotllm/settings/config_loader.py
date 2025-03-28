from pathlib import Path

from dynaconf import Dynaconf

SETTINGS_DIR = Path(__file__).parent

toml_files = list(SETTINGS_DIR.glob("**/*.toml"))

global_settings = Dynaconf(
    envvar_prefix=False,
    merge_enabled=True,
    load_dotenv=True,
    settings_files=toml_files,
)


def get_settings():
    return global_settings
