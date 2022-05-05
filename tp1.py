#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.layers import Conv2DTranspose, UpSampling2D, SeparableConv2D
from tensorflow.keras import layers

from tp1_utils import load_data, overlay_masks


def plot(hist):
    plt.figure(figsize=(12, 8))
    plt.plot(hist.history['loss'], label='train_loss', linewidth=2.0)
    plt.plot(hist.history['val_loss'], label='val_loss', linewidth=2.0)
    plt.plot(hist.history['accuracy'], label='train_acc', linewidth=2.0)
    plt.plot(hist.history['val_accuracy'], label='val_acc', linewidth=2.0)
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.margins(x=0)
    plt.margins(y=0)
    plt.legend()
    plt.show()


def build_segmentation_model():
    inputs = Input(shape=(64, 64, 3))

    ### [First half of the network: downsampling inputs] ###

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

    ### [Second half of the network: upsampling inputs] ###

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

    # Add a per-pixel classification layer
    outputs = Conv2D(2, 3, activation="softmax", padding="same")(x)

    # Define the model
    return Model(inputs, outputs)


def build_multilabel_model():
    inputs = Input(shape=(64, 64, 3), name='inputs')

    layer = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(128, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(128, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(128, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    features = Flatten(name='features')(layer)
    layer = Dense(256, activation="relu")(features)
    layer = Dropout(0.3)(layer)
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
    layer = Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3, 3), padding="same", activation="relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    features = Flatten(name='features')(layer)
    layer = Dense(256, activation="relu")(features)
    layer = Dropout(0.5)(layer)
    layer = Dense(10, activation="softmax")(layer)

    return Model(inputs=inputs, outputs=layer)


def multiclass_model(train_x, train_classes, test_x, test_classes):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_classes, test_size=500)

    EPOCHS = 100
    BATCH_SIZE = 32
    INIT_LR = 0.01
    MOMENTUM = 0.9

    model = build_multiclass_model()

    opt = SGD(learning_rate=INIT_LR, momentum=MOMENTUM, nesterov=True, decay=INIT_LR / EPOCHS)

    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)

    final_training_data = [(l, history.history[l][-1]) for l in history.history]
    print(final_training_data)

    loss, acc = model.evaluate(test_x, test_classes)
    print('Multiclass test loss:', loss)
    print('Multiclass test accuracy:', acc)

    plot(history)


def multilabel_model(train_x, train_labels, test_x, test_labels):
    train_X, val_x, train_y, val_y = train_test_split(train_x, train_labels, test_size=500)

    EPOCHS = 100
    BATCH_SIZE = 32
    INIT_LR = 0.01
    MOMENTUM = 0.9

    model = build_multilabel_model()

    opt = SGD(learning_rate=INIT_LR, momentum=MOMENTUM, nesterov=True, decay=INIT_LR / EPOCHS)

    model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_X, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)

    final_training_data = [(l, history.history[l][-1]) for l in history.history]
    print(final_training_data)

    loss, acc = model.evaluate(test_x, test_labels)
    print('Multilabel test loss:', loss)
    print('Multilabel test accuracy:', acc)

    plot(history)


def segmentation_model(train_x, train_masks, test_x, test_masks):
    train_X, val_x, train_y, val_y = train_test_split(train_x, train_masks, test_size=500)

    EPOCHS = 15
    BATCH_SIZE = 32
    INIT_LR = 0.01
    MOMENTUM = 0.9

    model = build_segmentation_model()

    opt = SGD(learning_rate=INIT_LR, momentum=MOMENTUM, nesterov=True, decay=INIT_LR / EPOCHS)

    model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_X, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)

    final_training_data = [(l, history.history[l][-1]) for l in history.history]
    print(final_training_data)

    loss, acc = model.evaluate(test_x, test_masks)
    print('Segmentation test loss:', loss)
    print('Segmentation test accuracy:', acc)

    predicts = model.predict(test_x)
    overlay_masks('test_overlay.png', test_x, predicts)

    plot(history)


def main():
    train_x, test_x, train_masks, test_masks, train_classes, train_labels, test_classes, test_labels = load_data().values()

    # multiclass_model(train_x, train_classes, test_x, test_classes)

    multilabel_model(train_x, train_labels, test_x, test_labels)

    # segmentation_model(train_x, train_masks, test_x, test_masks)


if __name__ == '__main__':
    main()
