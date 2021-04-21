# -*- coding: utf-8 -*-
# File: config_parser.py

r"""Json config parser, support attributes set, get and freeze.
Usage: from config_parser import config
    config.update_args(json_file)
    config.freeze()
"""

from __future__ import print_function

import os
import logger
import json

__all__ = ["config"]


def json_loader(json_file):
    """fast json loader"""
    if not os.path.exists(json_file):
        raise IOError("Json file path {} not exist.".format(json_file))
    try:
        import simplejson as json

        assert json
    except ImportError:
        import json
    with open(json_file) as fh:
        try:
            info = json.load(fh)
        except BaseException:
            raise ValueError("Invalid json format, please check your config.")
    return info


class AttrDict:
    """Attribute class wrapper."""

    _freezed = False

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed:
            # if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! not allowed to set key: {}".format(name)
            )
        super().__setattr__(name, value)

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {
            k: v.to_dict() if isinstance(v, AttrDict) else v
            for k, v in self.__dict__.items()
        }

    def from_dict(self, d):
        """Convert from nested dict."""
        self.freeze(False)
        for k, v in d.items():
            self_v = getattr(self, k)
            if isinstance(self_v, AttrDict):
                self_v.from_dict(v)
            else:
                setattr(self, k, v)

    def update_args(self, json_info):
        """Update from json dict. """
        for k, v in json_info.items():
            dic = self
            assert k in dir(dic), "Unknown config key: {}".format(k)
            setattr(dic, k, v)

    def freeze(self, freezed=True):
        """Freeze config values"""
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config  # short alias to avoid coding

# mode flags ---------------------
_C.MODEL_NAME = "test"  # string, model name.
_C.MODEL_VERSION = 1  # int, model version.
_C.APP_NAME = "process"  # string, application name.
_C.MODEL_PATH = "/tmp/model/"  # string, model path.
_C.RESOURCE_PATH = "/tmp/resource"  # string, customed resource path.
_C.USER_PARAMS = {"key": "value"}  # dict, user-defined params.


if __name__ == "__main__":
    logger.set_logger_dir("./log")
    tmp_json_file = "./sample.json"
    test_info = {
        "MODEL_NAME": "pickle",
        "APP_NAME": "tf_serving",
        "MODEL_PATH": "/this/is/a/model/path",
        "RESOURCE_PATH": "/this/is/a/resource/path",
        "USER_PARAMS": {"k1": 10, "k2": {"k3": ["a", "b"], "k4": 101}},
    }
    test_info_str = json.dumps(test_info, indent=4)
    with open(tmp_json_file, "w") as fw:
        fw.write(test_info_str)

    logger.info("----------------Initial Config----------------------\n" + str(_C))

    config.update_args(tmp_json_file)
    logger.info("----------------Updated Config---------------\n" + str(config))

    # finalize
    config.freeze()

    # clean
    os.remove(tmp_json_file)
