import logging
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from pydantic import BaseSettings, BaseModel, Json
from rq import Queue
import redis as rd


class Settings(BaseSettings):
    ready_to_train = False
    classes: List[str] = []
    num_images: int = 0
    extensions: List[str] = []


class TrainingException(Exception):
    pass


class ModelData(BaseModel):
    model_structure: str
    loss_function: str
    optimizer: str
    n_epochs: int
    seed: int = 123
    split: float = 0.2
    batch_size: int = 32


redis_instance = rd.Redis(host='dataset_redis', port=6381)
training_queue = Queue("training", connection=redis_instance)
logger = logging.getLogger("api")
dataset_settings = Settings()

connected = False
shutdown = False
pool = ThreadPoolExecutor(10)
WAIT_TIME = 10
