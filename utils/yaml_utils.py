import yaml
try:
    from addict import Dict
except ModuleNotFoundError:
    class Dict(dict):
        def __init__(self, mapping=None, **kwargs):
            super().__init__()
            mapping = mapping or {}
            for key, value in dict(mapping, **kwargs).items():
                self[key] = self._wrap(value)

        @classmethod
        def _wrap(cls, value):
            if isinstance(value, dict) and not isinstance(value, Dict):
                return cls(value)
            if isinstance(value, list):
                return [cls._wrap(item) for item in value]
            return value

        def __getattr__(self, key):
            if key not in self:
                self[key] = Dict()
            return self[key]

        def __setattr__(self, key, value):
            self[key] = self._wrap(value)
from ruamel.yaml import YAML
import ruamel.yaml as ryaml
import shutil
import os
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
    
def _convert_option_value(current_value, value):
    if isinstance(current_value, bool):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean option value: {value!r}")
    return type(current_value)(value)


def update_config_from_options(config, options):
    for option in options:
        key, value = option.split('=', 1)
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = _convert_option_value(d[keys[-1]], value)
    return config


from ruamel.yaml import YAML

def change_yaml_by_options(yaml_path, options):
    Yaml = YAML(typ='rt') 
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = Yaml.load(file)  

    for option in options:
        key, value = option.split('=', 1)
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = _convert_option_value(d[keys[-1]], value)

    with open(yaml_path, 'w', encoding='utf-8') as file:
        Yaml.dump(config, file) 



def save_yaml(args,yaml_path,options):
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(yaml_path))
    shutil.copyfile(yaml_path,dst_path)
    if options != None:
        change_yaml_by_options(dst_path,options)
