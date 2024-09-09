import pandas as pd
import deepchem as dc
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

# Concatenate the features and target for each set
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

base_dir = "dataset\\split"
train_file = os.path.join(base_dir, "train.csv")
test_file = os.path.join(base_dir, "test.csv")

# Create the directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

loader = dc.molnet.load_delaney(
    featurizer='Raw', split='index')
train_dataset = loader.create_dataset(train_file)
test_dataset = loader.create_dataset(test_file)

char_dict, length = dc.models.TextCNNModel.build_char_dict(train_file)

dc.models.TextCNNModel(ntasks=1,
                       seq_length=length)