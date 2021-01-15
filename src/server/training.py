import os

import requests
from secrets import API_KEY

from src.server.dependency import logger
import tensorflow as tf


def train_model(training_id, model_structure, loss_function, optimizer, n_epochs):
    logger.debug('[Training] Starting to train model ID: ' + training_id)

    dataset_root = '/app/src/public_dataset'

    img_height = 28
    img_width = 28
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_root,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_root,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    autotune_buf_size = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune_buf_size)
    validation_ds = validation_ds.cache().prefetch(buffer_size=autotune_buf_size)

    model = tf.keras.models.model_from_json(model_structure)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=validation_ds, epochs=n_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    logger.debug('[Training] Completed training on model ID: ' + training_id)

    result = {
        'training_accuracy': acc[-1],
        'validation_accuracy': val_acc[-1],
        'training_loss': loss[-1],
        'validation_loss': val_loss[-1]
    }

    # Send HTTP request to server with the statistics on this training

    headers = {
        'api_key': API_KEY
    }
    dataset_name = os.getenv('DATASET_NAME')
    server_port = os.getenv('SERVER_PORT')

    r = requests.post(
        'http://host.docker.internal:' + str(server_port) + '/training/result',
        headers=headers,
        json={
            'dataset_name': dataset_name,
            'training_id': training_id,
            'results': result
        })
    r.raise_for_status()

    print("[Training Results] Sent training results to server.")
