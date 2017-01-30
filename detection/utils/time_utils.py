import time

from detection.utils.logger import logger


def time_it(func):
    """
    A decorator for timing the execution time of functions.
    """

    def decorator(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info("Execution time : {}() = {}sec".format(func.__name__, end_time - start_time))
        return result

    return decorator
