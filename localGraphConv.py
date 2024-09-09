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

# Save the data to CSV files
train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

# Load and Featurize the data using DeepChem
tasks = ["single-class-label"]
ntasks = len(tasks)
# featurizer_func = dc.feat.ConvMolFeaturizer()
# loader = dc.data.CSVLoader(tasks=tasks, feature_field='SMILES', featurizer=featurizer_func)

train_dataset = train_file
test_dataset = test_file

char_dict, length = dc.models.TextCNNModel.build_char_dict(train_dataset)

# Batch size of models
batch_size = 64

# Featurize the SMILES strings using the SmilesToSeq featurizer
featurizer = dc.model.SmilesToSeq(seq_length=100)
train_dataset = dc.data.CSVLoader(
    tasks=[y_column],
    smiles_field=smiles_column,
    featurizer=featurizer
).create_dataset(train_file)

test_dataset = dc.data.CSVLoader(
    tasks=[y_column],
    smiles_field=smiles_column,
    featurizer=featurizer
).create_dataset(test_file)


# Initialize model
model = dc.models.TextCNNModel(
    n_tasks=1,
    batch_size=32,
    learning_rate=0.001,
    char_dict,
    seq_length=length,
    mode='classification',
    learning_rate=1e-3,
    batch_size=batch_size,
    use_queue=False,
)


# Train model
model.fit(train_dataset)

# Evaluate model
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score),
           dc.metrics.Metric(dc.metrics.f1_score),
           dc.metrics.Metric(dc.metrics.roc_auc_score)]

for metric in metrics:
    print("Train", metric.name, ":", model.evaluate(train_dataset, [metric]))
    print("Test", metric.name, ":", model.evaluate(test_dataset, [metric]))