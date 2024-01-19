import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def prepare_dataset1(sample_size=None, keep_all_features=False, separate=False):
    """
    Prepare dataset for Drug-Drug Interaction (DDI) prediction from IntegratedDS1.

    Parameters:
        sample_size (int or None): Number of samples to randomly select from the dataset. If None, use the entire dataset.
        keep_all_features (bool): Whether to keep all drug features or only those related to selected samples.
        separate (bool): [Unused] Parameter included for consistency with other functions.

    Returns:
        features_array (numpy array): Array containing pairs of drug features for DDI prediction.
        labels (numpy array): Flatten array of DDI labels corresponding to features_array.
    """
    drugs_features = pd.read_csv("DS1/IntegratedDS1.txt", delimiter=",", header=None, dtype=np.float32)
    drugs_interactions = pd.read_csv("DS1/drug_drug_matrix.csv", delimiter=",", header=None, dtype=bool)

    # Sample the dataset if specified
    if sample_size is not None and sample_size < drugs_interactions.shape[0]:
        drugs_interactions = drugs_interactions.sample(sample_size, random_state=42)
        drugs_features = drugs_features.iloc[drugs_interactions.index]

        # Filter drug features based on selected samples
        if not keep_all_features:
            drugs_interactions = drugs_interactions[
                drugs_interactions.columns[drugs_interactions.columns.isin(drugs_interactions.index)]]
            drugs_features = drugs_features[drugs_features.columns[drugs_features.columns.isin(drugs_features.index)]]

    # Create pairs of drug features for DDI prediction
    features_array = []
    for i in drugs_interactions.index:
        drug_features1 = list(drugs_features[i])
        for j in drugs_interactions.index:
            drug_features2 = list(drugs_features[j])
            features_array.append((drug_features1, drug_features2))

    return np.array(features_array), drugs_interactions.values.flatten()


def prepare_dataset2(sample_size=None, keep_all_features=False, separate=False):
    """
    Prepare dataset for Drug-Drug Interaction (DDI) prediction from DS2.

    Parameters:
        sample_size (int or None): Number of samples to randomly select from the dataset. If None, use the entire dataset.
        keep_all_features (bool): Whether to keep all drug features or only those related to selected samples.
        separate (bool): [Unused] Parameter included for consistency with other functions.

    Returns:
        features_array (numpy array): Array containing pairs of drug features for DDI prediction.
        labels (numpy array): Flatten array of DDI labels corresponding to features_array.
    """
    with open('DS2/ddiMatrix.csv') as x:
        ncols = len(x.readline().split(','))

    drugs_features = pd.read_csv("DS2/simMatrix.csv", delimiter=",", header=None, skiprows=[0], dtype=np.float16,
                                 usecols=range(1, ncols))
    drugs_interactions = pd.read_csv("DS2/ddiMatrix.csv", delimiter=",", header=None, skiprows=[0], dtype=bool,
                                     usecols=range(1, ncols))

    drugs_features = drugs_features.set_axis(range(drugs_features.shape[1]), axis=1)
    drugs_interactions = drugs_interactions.set_axis(range(drugs_interactions.shape[1]), axis=1)

    if sample_size is not None and sample_size < drugs_interactions.shape[0]:
        drugs_interactions = drugs_interactions.sample(sample_size, random_state=42)
        drugs_features = drugs_features.iloc[drugs_interactions.index]

        if not keep_all_features:
            drugs_interactions = drugs_interactions[
                drugs_interactions.columns[drugs_interactions.columns.isin(drugs_interactions.index)]]
            drugs_features = drugs_features[
                drugs_features.columns[drugs_features.columns.isin(drugs_features.index)]]

    features_array = []
    for i in drugs_interactions.index:
        drug_features1 = list(drugs_features[i])
        for j in drugs_interactions.index:
            drug_features2 = list(drugs_features[j])
            features_array.append((drug_features1, drug_features2))

    return np.array(features_array), drugs_interactions.values.flatten()


def preprocess_labels(labels, encoder=None, categorical=True):
    """
    Preprocess labels using given encoder and convert to categorical format if specified.

    Parameters:
        labels (numpy array or pandas Series): Array or Series of labels to be encoded.
        encoder (LabelEncoder or None): Existing encoder to be used, or None to create a new one.
        categorical (bool): Whether to convert labels to categorical format.

    Returns:
        y (numpy array): Encoded and optionally categorical labels.
        encoder (LabelEncoder): Encoder used for label transformation.
    """
    if not encoder:
        encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int8)

    if categorical:
        y = to_categorical(y)

    return y, encoder
