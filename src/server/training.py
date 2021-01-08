from keras_preprocessing.image import ImageDataGenerator

from src.server import dependency
from src.server.dependency import logger, dataset_settings
import tensorflow as tf


def train_model(training_id):
    logger.debug('[Training] Starting to train model')

    dataset_root = '/app/src/public_dataset'
    image_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)

    img_height = 250
    img_width = 250

    train_gen = image_generator.flow_from_directory(
        directory=str(dataset_root),
        shuffle=True,
        target_size=(img_height, img_width),
        classes=dataset_settings.classes,
        class_mode='binary',
        subset='training'
    )

    test_gen = image_generator.flow_from_directory(
        directory=str(dataset_root),
        shuffle=True,
        target_size=(img_height, img_width),
        classes=dataset_settings.classes,
        class_mode='binary',
        subset='validation'
    )

    dataset_settings.train_generator = train_gen
    dataset_settings.test_generator = test_gen
    logger.debug('Dataset Settings Updated')

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(250, 250)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5749)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(dataset_settings.train_generator, epochs=10)

    test_loss, test_acc = model.evaluate(dataset_settings.test_generator, verbose=2)

    logger.debug('[Training] Model Training Complete')

    return training_id, test_loss, test_acc
