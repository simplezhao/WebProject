# -*- coding: utf-8 -*-
# File: logger.py

"""
The logger module itself has the common logging functions of Python's logging.Logger.
For example:
    import logger
    logger.set_logger_dir('train_log/test')
    logger.info("Test")
    logger.error("Error happened!")
"""


import errno
import logging
import os
import os.path
import shutil
import sys
import time
import random
import string
from datetime import datetime
from termcolor import colored

__all__ = ["set_logger_dir", "auto_set_dir", "get_logger_dir"]


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored("[%(asctime)s @%(filename)s:%(lineno)d]", "green")
        msg = "%(message)s"
        if record.levelno == logging.WARNING:
            fmt = date + " " + colored("WRN", "red", attrs=["blink"]) + " " + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = (
                date
                + " "
                + colored("ERR", "red", attrs=["blink", "underline"])
                + " "
                + msg
            )
        elif record.levelno == logging.DEBUG:
            fmt = date + " " + colored("DBG", "yellow", attrs=["blink"]) + " " + msg
        else:
            fmt = date + " " + msg
        if hasattr(self, "_style"):
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


def _getlogger():
    # this file is synced to "dataflow" package as well
    package_name = "http_serving"
    logger = logging.getLogger(package_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt="%m%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = [
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    "debug",
    "setLevel",
    "addFilter",
]

# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)
# 'warn' is deprecated in logging module
warn = _logger.warning
__all__.append("warn")


def _get_time_str():
    return datetime.now().strftime("%m%d-%H%M%S")


# globals: logger file and directory
LOG_DIR = None
_FILE_HANDLER = None


def mkdir_p(dirname):
    """Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == "" or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def _set_file(path):
    global _FILE_HANDLER
    if os.path.isfile(path):
        backup_name = path + "." + _get_time_str()
        shutil.move(path, backup_name)
        _logger.info(
            "Existing log file '{}' backuped to '{}'".format(path, backup_name)
        )
    hdl = logging.FileHandler(filename=path, encoding="utf-8", mode="w")
    hdl.setFormatter(_MyFormatter(datefmt="%m%d %H:%M:%S"))

    _FILE_HANDLER = hdl
    _logger.addHandler(hdl)
    _logger.info("Argv: " + " ".join(sys.argv))


def set_logger_dir(dirname):
    """
    Set the directory for global logging.

    Args:
        dirname(str): log directory

    """
    dirname = os.path.normpath(dirname)
    global LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete
        # the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER
    LOG_DIR = dirname
    mkdir_p(dirname)
    rnd_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))
    _set_file(os.path.join(dirname, "log{}_{}.log".format(int(time.time()), rnd_str)))


def auto_set_dir(name=None):
    """
    Use :func:`logger.set_logger_dir` to set log directory to
    "./train_log/{scriptname}:{name}". "scriptname" is the name of the main python file currently running"""
    mod = sys.modules["__main__"]
    basename = os.path.basename(mod.__file__)
    auto_dirname = os.path.join("train_log", basename[: basename.rfind(".")])
    if name:
        auto_dirname += "_%s" % name if os.name == "nt" else ":%s" % name
    set_logger_dir(auto_dirname)


def get_logger_dir():
    """
    Returns:
        The logger directory, or None if not set.
    """
    return LOG_DIR
