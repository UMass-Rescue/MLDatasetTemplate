import os

import requests
from secrets import API_KEY

from src.server.dependency import ModelData
import tensorflow as tf


def train_model(training_id, model_data: ModelData):
    acc = [-1]
    val_acc = [-1]

    loss = [-1]
    val_loss = [-1]
    try:
        print('[Training] Starting to train model ID: ' + training_id)

        dataset_root = '/app/src/public_dataset'

        img_height = 28
        img_width = 28

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_root,
            validation_split=model_data.split,
            subset="training",
            seed=model_data.seed,
            image_size=(img_height, img_width),
            batch_size=model_data.batch_size
        )

        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_root,
            validation_split=model_data.split,
            subset="validation",
            seed=model_data.seed,
            image_size=(img_height, img_width),
            batch_size=model_data.batch_size
        )

        autotune_buf_size = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune_buf_size)
        validation_ds = validation_ds.cache().prefetch(buffer_size=autotune_buf_size)

        model = tf.keras.models.model_from_json(model_data.model_structure)

        model.compile(optimizer=model_data.optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(train_ds, validation_data=validation_ds, epochs=model_data.n_epochs)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print('[Training] Completed training on model ID: ' + training_id)
    except:
        print('[Training] Critical error on training: ' + training_id)

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
