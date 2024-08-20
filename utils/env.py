import os

def get_env_var(var, default=''):
    if var in os.environ:
        return os.environ[var]
    return default
