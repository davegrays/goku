import logging
import requests
import time
import glob
from datetime import timedelta
from functools import wraps
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image

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
            logger.getChild(func.__name__).log(
                time_log_level, f"Completed in {duration}."
            )
            return return_value

        return wrapper

    return decorator


def get_soup(url: str):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup


# @log_time(log)
def download_image(folder_path: str, url: str):
    r = requests.get(url)
    out_path = Path(folder_path) / Path(url).name
    with open(str(out_path), "wb") as f:
        f.write(r.content)


def downsample_images(in_folder_path: str, out_folder_path: str):
    file_paths = glob.glob(in_folder_path + "/*.jpg")
    for file_path in file_paths:
        with open(file_path, "rb") as file:
            img = Image.open(file)
            img = img.resize((img.width // 2, img.height // 2), Image.ANTIALIAS)
            filename = Path(file_path).name
            img.save(Path(out_folder_path) / filename)
