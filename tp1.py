#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, \
    Conv2DTranspose, GlobalAveragePooling2D, Concatenate
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


def double_conv_block(layer, n_filters):
    layer = Conv2D(n_filters, 3, padding="same", activation="relu")(layer)
    layer = Conv2D(n_filters, 3, padding="same", activation="relu")(layer)
    return layer


def downsample_block(layer, n_filters):
    feature = double_conv_block(layer, n_filters)
    pooling = MaxPooling2D(2)(feature)
    return feature, pooling


def upsample_block(layer, conv_features, n_filters):
    layer = Conv2DTranspose(n_filters, 3, 2, padding="same")(layer)
    layer = Concatenate([layer, conv_features])
    layer = double_conv_block(layer, n_filters)
    return layer


def build_segmentation_model(filters=None):
    if filters is None:
        filters = [32]

    inputs = Input(shape=(64, 64, 3))

    layer = inputs
    features = []
    for i, n_filters in enumerate(filters):
        feature, layer = downsample_block(layer, n_filters)
        features[i] = feature

    layer = double_conv_block(layer, 2 * filters[-1])

    for i, n_filters in reversed(list(enumerate(filters))):
        layer = upsample_block(layer, features[i], n_filters)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(layer)

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

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=multilabel_metrics())

    model.summary()

    fit_evaluate(model, "Semantic Segmentation", train_x, train_masks, test_x, test_masks, batch_size=32, epochs=20)

    predicts = model.predict(test_x)
    overlay_masks('test_overlay.png', test_x, predicts)
    compare_masks('test_compare.png', test_masks, predicts)


def build_transfer_model(output_activation):
    inputs = Input(shape=(64, 64, 3), name='inputs')
    convolutional_base = EfficientNetB0(include_top=False, weights='imagenet')
    layer = convolutional_base(inputs, training=False)
    convolutional_base.trainable = False

    layer = GlobalAveragePooling2D()(layer)

    layer = Dense(10, activation=output_activation)(layer)

    return Model(inputs=inputs, outputs=layer)


def transfer_multilabel_model(train_x, train_labels, test_x, test_labels):
    model = build_transfer_model("sigmoid")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    fit_evaluate(model, "Multilabel", train_x * 255, train_labels, test_x * 255, test_labels, batch_size=32, epochs=100)


def transfer_multiclass_model(train_x, train_classes, test_x, test_classes):
    model = build_transfer_model("softmax")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    fit_evaluate(model, "Multiclass", train_x * 255, train_classes, test_x * 255, test_classes, batch_size=32,
                 epochs=100)


def main():
    train_x, test_x, train_masks, test_masks, train_classes, train_labels, test_classes, test_labels = load_data().values()

    multiclass_model(train_x, train_classes, test_x, test_classes)

    transfer_multiclass_model(train_x, train_classes, test_x, test_classes)

    multilabel_model(train_x, train_labels, test_x, test_labels)

    transfer_multilabel_model(train_x, train_labels, test_x, test_labels)

    segmentation_model(train_x, train_masks, test_x, test_masks)


if __name__ == '__main__':
    main()
