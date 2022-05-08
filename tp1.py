#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, \
    Dropout, Conv2DTranspose, UpSampling2D, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tp1_utils import load_data, overlay_masks


def show_metrics(history, model, metrics):
    print(f"{model}:")
    for metric_label in metrics:
        print(f"{metric_label.capitalize()} {metrics[metric_label]:.5f}")
    plot(history, model)


def plot(hist, model):
    plt.figure(figsize=(12, 8))
    for k in hist.history:
        plt.plot(hist.history[k], label=k, linewidth=2.0)

    plt.title('Training and Validation Metrics: ' + model)
    plt.xlabel('Epoch #')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_segmentation_model():
    inputs = Input(shape=(64, 64, 3))

    # Downsampling inputs

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Upsampling inputs

    for filters in [64, 32]:
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Pixelwise classification layer
    outputs = Conv2D(2, 3, activation="softmax", padding="same")(x)

    return Model(inputs, outputs)


def build_multilabel_model():
    inputs = Input(shape=(64, 64, 3), name='inputs')

    layer = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    layer = GlobalAveragePooling2D()(layer)
    # layer = Flatten(name='features')(layer)
    # layer = Dense(256, activation="relu")(layer)
    # layer = Dropout(0.3)(layer)
    layer = Dense(10, activation="sigmoid")(layer)

    return Model(inputs=inputs, outputs=layer)


def build_multiclass_model():
    inputs = Input(shape=(64, 64, 3), name='inputs')

    layer = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    layer = Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    # layer = Flatten(name='features')(layer)
    # layer = Dense(256, activation="relu")(layer)
    # layer = Dropout(0.5)(layer)
    layer = GlobalAveragePooling2D()(layer)
    layer = Dense(10, activation="softmax")(layer)

    return Model(inputs=inputs, outputs=layer)


def multiclass_model(train_x, train_classes, test_x, test_classes):
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_classes, test_size=500)

    EPOCHS = 20
    BATCH_SIZE = 32

    model = build_multiclass_model()

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=["accuracy"])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

    metrics = model.evaluate(test_x, test_classes, verbose=0, return_dict=True)

    show_metrics(history, model, metrics)


def multilabel_model(train_x, train_labels, test_x, test_labels):
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_labels, test_size=500)

    EPOCHS = 100
    BATCH_SIZE = 32

    model = build_multilabel_model()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=multilabel_metrics())

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

    metrics = model.evaluate(test_x, test_labels)

    show_metrics(history, "Multilabel", metrics)


def multilabel_metrics(thr=0.5):
    return [
        metrics.BinaryAccuracy(name="accuracy", threshold=thr),
        metrics.Precision(thresholds=thr),
        metrics.Recall(thresholds=thr),
        # metrics.TruePositives(thresholds=thr),
        # metrics.TrueNegatives(thresholds=thr),
        # metrics.FalsePositives(thresholds=thr),
        # metrics.FalseNegatives(thresholds=thr)
    ]


def segmentation_model(train_x, train_masks, test_x, test_masks):
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_masks, test_size=500)

    EPOCHS = 100
    BATCH_SIZE = 32

    model = build_segmentation_model()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

    metrics = model.evaluate(test_x, test_masks)

    show_metrics(history, "Semantic Segmentation", metrics)

    predicts = model.predict(test_x)
    overlay_masks('test_overlay.png', test_x, predicts)


def build_multiclass_transfer_model():
    convolutional_base = MobileNetV2(include_top=False, input_shape=(64, 64, 3), weights='imagenet')
    convolutional_base.trainable = False

    inputs = Input(shape=(64, 64, 3), name='inputs')
    layer = convolutional_base(inputs, training=False)

    # layer = GlobalAveragePooling2D()(layer)

    layer = Dropout(0.5)(layer)
    layer = Flatten(name='features')(layer)
    layer = Dense(64, activation="relu")(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(10, activation="softmax")(layer)

    return Model(inputs=inputs, outputs=layer)


def transfer_multiclass_model(train_x, train_classes, test_x, test_classes):
    train_X = preprocess_input(train_x * 255)
    train_X, val_x, train_y, val_y = train_test_split(train_X, train_classes, test_size=500)

    X_test = preprocess_input(test_x * 255)

    EPOCHS = 100
    BATCH_SIZE = 32

    model = build_multiclass_transfer_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_X, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)

    metrics = model.evaluate(X_test, test_classes)

    show_metrics(history, "Transfer Multiclass", metrics)


def main():
    train_x, test_x, train_masks, test_masks, train_classes, train_labels, test_classes, test_labels = load_data().values()

    multiclass_model(train_x, train_classes, test_x, test_classes)

    # transfer_multiclass_model(train_x, train_classes, test_x, test_classes)

    # multilabel_model(train_x, train_labels, test_x, test_labels)

    # segmentation_model(train_x, train_masks, test_x, test_masks)


if __name__ == '__main__':
    main()
