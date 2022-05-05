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

from tp1_utils import load_data


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


def build_multilabel_model():
    inputs = Input(shape=(64, 64, 3), name='inputs')

    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    features = Flatten(name='features')(layer)
    layer = Dense(512)(features)
    layer = Activation("relu")(layer)
    layer = Dense(256)(features)
    layer = Activation("relu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)

    return Model(inputs=inputs, outputs=layer)


def build_multiclass_model():
    inputs = Input(shape=(64, 64, 3), name='inputs')

    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)

    features = Flatten(name='features')(layer)
    layer = Dense(256)(features)
    layer = Activation("relu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)

    return Model(inputs=inputs, outputs=layer)


def multiclass_model(train_x, train_classes, test_x, test_classes):
    train_X, val_x, train_y, val_y = train_test_split(train_x, train_classes, test_size=500)

    model = build_multiclass_model()

    opt = SGD(learning_rate=INIT_LR, momentum=MOMENTUM, nesterov=True, decay=INIT_LR / EPOCHS)

    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_X, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)
    final_training_data = [(l, history.history[l][-1]) for l in history.history]

    print(final_training_data)
    print(100 - model.evaluate(test_x, test_classes)[1] * 100)

    plot(history)


def multilabel_model(train_x, train_labels, test_x, test_labels):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_labels, test_size=500)

    model = build_multilabel_model()

    opt = SGD(learning_rate=INIT_LR, momentum=MOMENTUM, nesterov=True, decay=INIT_LR / EPOCHS)

    model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)
    final_training_data = [(l, history.history[l][-1]) for l in history.history]

    print(final_training_data)
    print(100 - model.evaluate(test_x, test_labels)[1] * 100)

    plot(history)


def segmentation_model(train_x, train_masks, test_x, test_masks):
    return -1


if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 32
    INIT_LR = 0.01
    MOMENTUM = 0.9

    train_x, test_x, train_masks, test_masks, train_classes, train_labels, test_classes, test_labels = load_data().values()

    multiclass_model(train_x, train_classes, test_x, test_classes)

# multilabel_model(train_x, train_labels, test_x, test_labels)

# segmentation_model(train_x, train_masks, test_x, test_masks)
