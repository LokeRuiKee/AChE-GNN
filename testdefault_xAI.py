import pandas as pd
import deepchem as dc
from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from captum.attr import IntegratedGradients
import torch.nn.functional as F

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
featurizer_func = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=tasks, feature_field='SMILES', featurizer=featurizer_func)

train_dataset = loader.create_dataset(train_file)
test_dataset = loader.create_dataset(test_file)

# Initialize model
model = dc.models.GraphConvModel(n_tasks=ntasks, mode='classification', batch_normalize=False)

# Train model
model.fit(train_dataset)

# Evaluate model
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score),
           dc.metrics.Metric(dc.metrics.f1_score),
           dc.metrics.Metric(dc.metrics.roc_auc_score)]

for metric in metrics:
    print("Train", metric.name, ":", model.evaluate(train_dataset, [metric]))
    print("Test", metric.name, ":", model.evaluate(test_dataset, [metric]))

# Convert the DeepChem model to PyTorch model for Captum
dc_torch_model = model.model

# Explainability using Captum
ig = IntegratedGradients(dc_torch_model)

# Access a specific sample from the test_dataset
sample_index = 0  # Choose the index of the sample you want to explain
sample_data = test_dataset.select([sample_index])

# Convert the data from the sample to the correct format
sample_X = torch.tensor(np.concatenate(sample_data.X).astype(np.float32))
sample_w = torch.tensor(sample_data.w.astype(np.float32))

# Forward pass
pred_class = dc_torch_model(sample_X)
pred_class = F.softmax(pred_class, dim=1).argmax(dim=1).item()

# Get attribution using Integrated Gradients
attributions, delta = ig.attribute(sample_X.unsqueeze(0), target=pred_class, return_convergence_delta=True)

# The attributions are in the same shape as the input features
print("Attributions: ", attributions)

# Visualize or interpret the attributions
