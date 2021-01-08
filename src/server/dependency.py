import logging
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from pydantic import BaseSettings

logger = logging.getLogger("api")


class Settings(BaseSettings):
    ready_to_train = False
    classes: List[str] = []
    num_images: int = 0
    extensions: List[str] = []


dataset_settings = Settings()


class TrainingException(Exception):
    pass


connected = False
shutdown = False
pool = ThreadPoolExecutor(10)
WAIT_TIME = 10
