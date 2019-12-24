"""
Define functions used by this package generally.
"""
from typing import Optional
from functools import wraps
import os
import random
import time
import logging
import numpy as np

import torch

def set_seed(seed: Optional[int] = None, use_gpu: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        if use_gpu > 0:
            torch.cuda.manual_seed_all(seed)

def set_logger(
        log_path: str = 'logs/test.log',
        log_level: int = logging.INFO):

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s]: %(message)s")

    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    logging.root.setLevel(log_level)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stdout_handler)

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time =  time.time() - start
        print('{} は {} 秒かかりました'.format(func.__name__, elapsed_time))

        return result

    return wrapper
