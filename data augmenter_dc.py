import pandas as pd
from sklearn.model_selection import train_test_split
import os
from rdkit import Chem
import deepchem as dc
import numpy as np

# Load the dataset
file_path = "dataset\\Human_dataset_1micromolar.xlsx"
df = pd.read_excel(file_path)

# Specify the columns
smiles_column = "SMILES"
y_column = "single-class-label"

# Split the data into features and target
X = df[smiles_column]
y = df[y_column]

# Split the data into training and test sets (80%, 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to augment SMILES
def augment_smiles(smiles, n_augmentations=5):
    mol = Chem.MolFromSmiles(smiles)
    smiles_list = [Chem.MolToSmiles(mol, doRandom=True) for _ in range(n_augmentations)]
    return smiles_list

# Augment SMILES for training data
augmented_smiles = []
augmented_labels = []
for smiles, label in zip(X_train, y_train):
    augmented_smiles.extend(augment_smiles(smiles))
    augmented_labels.extend([label] * 5)

# Convert augmented SMILES to DeepChem graphs
featurizer = dc.feat.ConvMolFeaturizer()
train_graphs = featurizer.featurize(augmented_smiles)
train_labels = np.array(augmented_labels)

# Filter out None values
train_graphs, train_labels = zip(*[(graph, label) for graph, label in zip(train_graphs, train_labels) if graph is not None])

# Create DeepChem NumpyDataset for training data
train_dataset = dc.data.NumpyDataset(np.array(train_graphs), np.array(train_labels))

# Repeat the process for test data without augmentation
test_graphs = featurizer.featurize(X_test)
test_labels = np.array(y_test)

# Filter out None values
test_graphs, test_labels = zip(*[(graph, label) for graph, label in zip(test_graphs, test_labels) if graph is not None])

# Create DeepChem NumpyDataset for test data
test_dataset = dc.data.NumpyDataset(np.array(test_graphs), np.array(test_labels))

# Save the processed data
base_dir = "dataset\\split"
train_file = os.path.join(base_dir, "train_dataset.pkl")
test_file = os.path.join(base_dir, "test_dataset.pkl")

# Create the directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)


loader = dc.data.CSVLoader(tasks=["single-class-label"], feature_field="SMILES", featurizer=featurizer)
dataset = loader.create_dataset(train_dataset, train_file)
dataset = loader.create_dataset(test_dataset, test_file)