# Load configuration
from pathlib import Path

import yaml

config_path = Path(__file__).with_name("config.yaml")

with config_path.open("r") as f:
    config = yaml.safe_load(f)
    print(config)
