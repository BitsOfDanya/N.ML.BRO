import logging
import os


def init_logging(debug=False, file=None):
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler()]
    if file:
        dirname = os.path.dirname(file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        handlers.append(logging.FileHandler(file))
    logging.basicConfig(level=level, format='[%(levelname)s\t%(asctime)s] %(message)s',
                        handlers=handlers)
