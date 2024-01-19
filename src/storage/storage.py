import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def save_data(x_train, x_test, y_train, y_test, path=""):
	if not os.path.exists(path):
		os.makedirs(path)
	# Save training data
	np.save(os.path.join(path, "X_train.npy"), x_train)
	np.save(os.path.join(path, "X_test.npy"), x_test)
	np.save(os.path.join(path, "y_train.npy"), y_train)
	np.save(os.path.join(path, "y_test.npy"), y_test)


def load_data(path=""):
	# Load training data
	train = (np.load(os.path.join(path, "X_train.npy"), mmap_mode='r'), np.load(os.path.join(path, "y_train.npy"), mmap_mode='r'))
	test = (np.load(os.path.join(path, "X_test.npy"), mmap_mode='r'), np.load(os.path.join(path, "y_test.npy"), mmap_mode='r'))

	return train[0], test[0], train[1], test[1]


def save_label_encoder(encoder, path=""):
	if not os.path.exists(path):
		os.makedirs(path)
	# Save LabelEncoder
	joblib.dump(encoder.classes_, os.path.join(path, "label_encoder.joblib"))


def load_label_encoder(path=""):
	# Load LabelEncoder
	classes = joblib.load(os.path.join(path, "label_encoder.joblib"))
	encoder = LabelEncoder()
	encoder.classes_ = classes

	return encoder


def save_model_data(model, model_name, path=""):
	if not os.path.exists(path):
		os.makedirs(path)
	# Save the entire model
	model.save(os.path.join(path, f"{model_name}.h5"))


def load_model_data(model_name, path=""):
	# Load the entire model
	model = load_model(os.path.join(path, f"{model_name}.h5"))
	return model
