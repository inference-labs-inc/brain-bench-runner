import logging

logger = logging.getLogger(__name__)

def assert_success(function):
    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        if not isinstance(res, dict) and res is not True or res is None:
            logger.error("Response from failed function {}:\n {}".format(function.__name__, res))
            raise Exception("Function {} failed to execute successfully.".format(function.__name__))
    return wrapper
