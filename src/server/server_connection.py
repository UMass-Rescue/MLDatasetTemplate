import os

import requests
import time

from fastapi.logger import logger

from secrets import API_KEY
from src.server import dependency


def register_model_to_server(server_port, model_port, model_name):
    """
    Send notification to the server with the training name and port to register the microservice
    It retries until a connection with the server is established
    """
    while not dependency.shutdown:
        try:
            headers = {
                'api_key': API_KEY
            }
            r = requests.post('http://host.docker.internal:' + str(server_port) + '/training/register',
                              headers=headers,
                              json={"modelName": model_name, "modelPort": model_port})
            r.raise_for_status()
            dependency.connected = True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError):
            dependency.connected = False
            logger.debug('Registering to server fails. Retry in ' + str(dependency.WAIT_TIME) + ' seconds')

        # Delay for WAIT_TIME between server registration pings
        for increment in range(dependency.WAIT_TIME):
            if not dependency.shutdown:  # Check between increments to stop hanging on shutdown
                time.sleep(1)

    logger.debug("[Healthcheck] Server Registration Thread Halted.")


async def send_model_results_to_server(training_id, training_results):
    """
    Send machine learning results to server.
    """
    if not dependency.connected:
        print('Unable to send training results to server. Connection is not established.')
        return

    headers = {
        'api_key': API_KEY
    }
    model_name = os.getenv('NAME')
    server_port = os.getenv('SERVER_PORT')

    try:
        r = requests.post(
            'http://host.docker.internal:' + str(server_port) + '/training/result',
            headers=headers,
            json={
                'model_name': model_name,
                'training_id': training_id,
                'results': training_results
            })
        r.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError):
        print('Unable to send training results to server. Established connection unsuccessful.')
        return

    print("[Training Results] Sent training results to server.")
