import os

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
try:
    import tomllib

    with open(config_file, 'rb') as f:
        config = tomllib.load(f)
except ModuleNotFoundError:
    import toml

    with open(config_file, encoding='utf-8') as f:
        config = toml.load(f)
