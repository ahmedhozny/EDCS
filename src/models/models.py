from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense, Flatten
import numpy as np
from keras.optimizers import legacy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def neural_binary_model(input_dim):
	# Create a Sequential model
	model = Sequential()

	# Add a Dense layer with 400 units, input dimension specified, and ReLU activation
	model.add(Dense(units=400, input_dim=input_dim, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))  # Add dropout layer

	# Add another Dense layer with 300 units and ReLU activation
	model.add(Dense(units=300, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))  # Add dropout layer

	# Add the output layer with 2 units and sigmoid activation for binary classification
	model.add(Dense(units=2, kernel_initializer='glorot_normal'))
	model.add(Activation('sigmoid'))

	# Use the SGD optimizer with specified parameters
	sgd = legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	# Compile the model with binary cross-entropy loss and the SGD optimizer
	model.compile(loss='binary_crossentropy', optimizer=sgd)  # Assigned the optimizer instance

	return model


def k_nearest_neighbors():
	return KNeighborsClassifier(n_neighbors=21)


def linear_discriminant_analysis():
	return LinearDiscriminantAnalysis()


""" WORK IN PROGRESS """


def neural_binary_model2(input_shape):
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))  # Flatten the input matrix
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))


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
