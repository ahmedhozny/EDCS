from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
import tensorflow as tf
import numpy as np


def neural_binary_model(input_dim):
	model = Sequential()
	model.add(Dense(500, input_dim=input_dim, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dense(400, input_dim=600, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(300, input_dim=400, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, input_dim=300, kernel_initializer='glorot_normal'))
	model.add(Activation('sigmoid'))
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=0.01,
		decay_steps=10000,
		decay_rate=0.9)
	optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model


# WIP
def custom_fit(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
	num_samples = len(X_train)
	print(num_samples)
	num_batches = (num_samples - 1) // batch_size + 1

	X_train = np.array_split(X_train, batch_size)
	y_train = np.array_split(y_train, batch_size)

	split_index = int(num_samples * (1 - validation_split))
	X_train, X_val = X_train[:split_index], X_train[split_index:]
	y_train, y_val = y_train[:split_index], y_train[split_index:]

	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}/{epochs}:")
		for batch_index in range(num_batches):
			# Get a mini-batch with shuffled indices
			batch_x = X_train[batch_index]
			batch_y = y_train[batch_index]

			# Train the model on the mini-batch
			loss = model.train_on_batch(batch_x, batch_y)

			# Print or log training progress
			print(f"Batch {batch_index + 1}/{num_batches} - Loss: {loss}")

		# Evaluate validation loss after each epoch
		val_loss = model.evaluate(X_val, y_val, verbose=0)
		print(f"Validation Loss: {val_loss}")
