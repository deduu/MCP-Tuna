# Load configuration
from pathlib import Path
import yaml

with open("./AgentY/data_generator_pipeline/scripts/config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    print(config)
