import logging

DARK_RED = '\033[31m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class CustomFormatter(logging.Formatter):
    """ Custom formatter to add color to log levels. """
    def __init__(self, fmt='%(levelname)s: %(message)s', datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        if record.levelno == logging.ERROR:
            prefix = RED
        elif record.levelno == logging.CRITICAL:
            prefix = DARK_RED
        elif record.levelno == logging.WARNING:
            prefix = YELLOW
        else:
            prefix = RESET

        # Original format message
        original = super(CustomFormatter, self).format(record)

        return prefix + original + RESET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s: %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

for handler in logging.root.handlers:
    handler.setFormatter(CustomFormatter('%(asctime)s %(filename)s: %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
