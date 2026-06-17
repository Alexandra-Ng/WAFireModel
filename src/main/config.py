"""Load experimental settings from the YAML file into a dict"""

from pathlib import Path
import yaml

def load_config(path):
    return yaml.safe_load(Path(path).read_text())


