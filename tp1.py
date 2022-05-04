#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""
import numpy as np

from tp1_utils import load_data
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
	layer = Conv2D(64, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization()(layer)
	layer = MaxPooling2D(pool_size=(2, 2))(layer)

	features = Flatten(name='features')(layer)
	layer = Dense(256)(features)
	layer = Activation("relu")(layer)
	# layer = BatchNormalization()(layer)
	layer = Dropout(0.5)(layer)
	layer = Dense(10)(layer)
	layer = Activation("softmax")(layer)

	return Model(inputs=inputs, outputs=layer)


def main():
	EPOCHS = 10
	BATCH_SIZE = 32
	INIT_LR = 0.01
	MOMENTUM = 0.9

	train_x, test_x, _, _, train_classes, _, test_classes, _ = load_data().values()

	train_x, val_x, train_y, val_y = train_test_split(train_x, train_classes, test_size=500)

	model = build_multiclass_model()

	opt = SGD(learning_rate=INIT_LR, momentum=MOMENTUM, nesterov=True, decay=INIT_LR/EPOCHS)

	model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS)
	final_training_data = [(l, history.history[l][-1]) for l in history.history]

	print(final_training_data)

	plot(history)


main()
