import utils
import os
import yaml
import json
from model_wrapper.cnn_wrapper import CNNWrapper

class yaml_parser(object):
    def __init__(self, config):
        for k, v in config.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [yaml_parser(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, yaml_parser(v) if isinstance(v, dict) else v)

def get_config_tranmicro(CONFIG_PATH):
    # load config
    with open(os.path.join('./designspace/Trans_Micro/configs', CONFIG_PATH+'.yaml'), 'r') as f:
        config = utils.yaml_parser(yaml.unsafe_load(f))
    
    return config

def get_searchspace_tranmicro(config):
    with open(os.path.join('./designspace/Trans_Micro/', config.arch_config.train_archs_accs), 'r') as t:
        searchspace = json.load(t)

    return searchspace
