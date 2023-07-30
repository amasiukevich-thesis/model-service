import yaml
import os

import config

CONFIG_PATH = os.environ.get("CONFIG_PATH")

def parse_yaml():

    yaml_dict = {}
    with open(CONFIG_PATH, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("ERROR")
            print(exc)

    return yaml_dict


config_dict = parse_yaml()

MODEL_DIR = config_dict.get('MODEL_DIR')
MODEL_VERSION = config_dict.get('MODEL_VERSION')
MODEL_CLASS = config_dict.get('MODEL_CLASS')


N_FEATURES = config_dict.get('N_FEATURES')

