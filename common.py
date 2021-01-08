import warnings
import yaml

def warningless_yaml_load(string):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yaml.load(string)
