#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Activation, Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D, \
    Dropout, Conv2DTranspose, UpSampling2D, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tp1_utils import load_data, overlay_masks, compare_masks


def show_metrics(model_name, history, metrics):
    print(f"{model_name}:")
    for metric_label in metrics:
        print(f"{metric_label.capitalize()} {metrics[metric_label]:.5f}")
    plot(history, model_name)


def plot(hist, model_name):
    plt.figure(figsize=(12, 8))
    for metric in hist.history:
        plt.plot(hist.history[metric], label=metric, linewidth=2.0)

    plt.title('Training and Validation Metrics: ' + model_name)
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
    outputs = Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    return Model(inputs, outputs)


def build_convnet(output_activation):
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
    # layer = Dense(128, activation="relu")(layer)
    # layer = Dropout(0.5)(layer)
    layer = Dense(10, activation=output_activation)(layer)

    return Model(inputs=inputs, outputs=layer)


def fit_evaluate(model, model_name, train_x, train_y, test_x, test_y, batch_size, epochs):
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=500)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

    metrics = model.evaluate(test_x, test_y, verbose=0, return_dict=True)

    show_metrics(model_name, history, metrics)


def multiclass_model(train_x, train_classes, test_x, test_classes):
    model = build_convnet(output_activation="softmax")

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=["accuracy"])

    fit_evaluate(model, "Multiclass", train_x, train_classes, test_x, test_classes, batch_size=32, epochs=40)


def multilabel_model(train_x, train_labels, test_x, test_labels):
    model = build_convnet(output_activation="sigmoid")

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=multilabel_metrics())

    fit_evaluate(model, "Multilabel", train_x, train_labels, test_x, test_labels, batch_size=32, epochs=40)


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
    model = build_segmentation_model()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    fit_evaluate(model, "Semantic Segmentation", train_x, train_masks, test_x, test_masks, batch_size=32, epochs=200)

    predicts = model.predict(test_x)
    overlay_masks('test_overlay.png', test_x, predicts)
    compare_masks('test_compare.png', test_masks, predicts)


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

    show_metrics("Transfer Multiclass", history, metrics)


def main():
    train_x, test_x, train_masks, test_masks, train_classes, train_labels, test_classes, test_labels = load_data().values()

    multiclass_model(train_x, train_classes, test_x, test_classes)

    # transfer_multiclass_model(train_x, train_classes, test_x, test_classes)

    # multilabel_model(train_x, train_labels, test_x, test_labels)

    # segmentation_model(train_x, train_masks, test_x, test_masks)


if __name__ == '__main__':
    main()
