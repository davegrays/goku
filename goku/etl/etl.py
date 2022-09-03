import logging
import requests
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path
from bs4 import BeautifulSoup


time_log_level = logging.DEBUG + 5
logging.addLevelName(time_log_level, "TIMING")

# log = logging.getLogger(__name__)
# log.setLevel(level=logging.INFO)


def log_time(logger: logging.Logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            starttime = time.time()
            logger.getChild(func.__name__).log(time_log_level, "Starting...")
            return_value = func(*args, **kwargs)
            duration = timedelta(seconds=int(time.time() - starttime))
            logger.getChild(func.__name__).log(time_log_level, f"Completed in {duration}.")
            return return_value
        return wrapper
    return decorator


def get_soup(url: str):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


# @log_time(log)
def download_image(folder_path: str, url: str):
    r = requests.get(url)
    out_path = Path(folder_path) / Path(url).name
    with open(str(out_path), 'wb') as f:
        f.write(r.content)
