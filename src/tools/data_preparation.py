import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def prepare_dataset2(sample_size=None, keep_all_features=False, separate=False):
	# Gets data size
	with open('DS2/ddiMatrix.csv') as x:
		ncols = len(x.readline().split(','))

	drugs_features = pd.read_csv("DS2/simMatrix.csv", delimiter=",", header=None, skiprows=[0], dtype=np.float16, usecols=range(1, ncols))
	drugs_interactions = pd.read_csv("DS2/ddiMatrix.csv", delimiter=",", header=None, skiprows=[0], dtype=bool, usecols=range(1, ncols))

	# Set column names for better readability
	drugs_features = drugs_features.set_axis(range(drugs_features.shape[1]), axis=1)
	drugs_interactions = drugs_interactions.set_axis(range(drugs_interactions.shape[1]), axis=1)

	if sample_size is not None and sample_size < drugs_interactions.shape[0]:
		# Randomly chooses n elements from the dataframe
		drugs_interactions = drugs_interactions.sample(sample_size, random_state=42)
		drugs_features = drugs_features.iloc[drugs_interactions.index]

		# (Not sure if we should keep all drugs features)
		if not keep_all_features:
			drugs_interactions = drugs_interactions[
				drugs_interactions.columns[drugs_interactions.columns.isin(drugs_interactions.index)]]
			drugs_features = drugs_features[drugs_features.columns[drugs_features.columns.isin(drugs_features.index)]]

	features_array = []

	for i in drugs_interactions.index:
		drug_features = list(drugs_features[i])

		# Duplicate the data (What's the point of duplicating though?)
		if separate:
			tmp_fea = (drug_features, drug_features)
		else:
			tmp_fea = drug_features + drug_features

		features_array += [tmp_fea] * drugs_interactions.shape[1]

	return np.array(features_array), drugs_interactions.values.flatten()


def preprocess_labels(labels, encoder=None, categorical=True):
	# Encode labels using LabelEncoder
	if not encoder:
		encoder = LabelEncoder()
	encoder.fit(labels)
	y = encoder.transform(labels).astype(np.int8)

	# Convert labels to categorical format if specified
	if categorical:
		y = to_categorical(y)

	return y, encoder
