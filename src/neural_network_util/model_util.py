import json
import tensorflow as tf

import os, sys, psutil, time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


# Taken from AstroNet. They know this stuff works
def _maybe_convert_dict(value):
  if isinstance(value, dict):
    return ConfigDict(value)

  return value

# Permette di ottenere una classe personalizzata di config. Opera delle azioni di conversione dei valori (non so se necessari in realtà)

class ConfigDict(dict):
  """Configuration container class."""

  def __init__(self, initial_dictionary=None):
    """Creates an instance of ConfigDict.

    Args:
      initial_dictionary: Optional dictionary or ConfigDict containing initial
        parameters.
    """
    if initial_dictionary:
      for field, value in initial_dictionary.items():
        initial_dictionary[field] = _maybe_convert_dict(value)
    super(ConfigDict, self).__init__(initial_dictionary)

  def __setattr__(self, attribute, value):
    self[attribute] = _maybe_convert_dict(value)

  def __getattr__(self, attribute):
    try:
      return self[attribute]
    except KeyError as e:
      raise AttributeError(e)

  def __delattr__(self, attribute):
    try:
      del self[attribute]
    except KeyError as e:
      raise AttributeError(e)

  def __setitem__(self, key, value):
    super(ConfigDict, self).__setitem__(key, _maybe_convert_dict(value))


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