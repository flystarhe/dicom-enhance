import os
import time
import logging
import numpy as np
from collections import defaultdict


class Cache(object):

    def __init__(self):
        self.data = defaultdict(list)

    def keep(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)

    def summary(self, num):
        data = [(k, np.mean(self.data[k])) for k in sorted(self.data)]
        message = ['{}:{:.6f}'.format(k, v) for k, v in data]
        message = '{}/{}'.format(num, ','.join(message))
        self.data = defaultdict(list)
        return message


def get_root_logger(log_dir, level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=fmt, level=level)

        os.makedirs(log_dir, exist_ok=True)
        filename = time.strftime('%m%d.%H%M%S')
        log_file = os.path.join(log_dir, filename)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(fmt))
        file_handler.setLevel(level)

        logger.addHandler(file_handler)
    return logger
