import json
from neural_network_util.model_util import ConfigDict


def load_config(config_path):
    config = None
    try:
        with open(config_path) as f:
            config = json.load(f)
            f.close()
    except ValueError and IOError as e:
        print(e)

    config = ConfigDict(config)

    return config