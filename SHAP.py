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

print(model.model.summary())

import shap

# SHAP requires a function that returns the model's predictions
def shap_model_predict(featurized_data):
    return np.array(model.predict(featurized_data))[:, 0]  # Adjust for your model's output

# Switch to a simpler featurizer if needed
featurizer_func = dc.feat.RDKitDescriptors()

# Reload datasets with tabular featurizer
loader = dc.data.CSVLoader(tasks=tasks, feature_field='SMILES', featurizer=featurizer_func)
train_dataset = loader.create_dataset(train_file)
test_dataset = loader.create_dataset(test_file)

# Get the features for SHAP
X_train_featurized = train_dataset.X
X_test_featurized = test_dataset.X

print("New Shape of X_train_featurized:", X_train_featurized.shape)

# Re-run SHAP explanation
explainer = shap.KernelExplainer(shap_model_predict, X_train_featurized[:100])
shap_values = explainer.shap_values(X_test_featurized[0].reshape(1, -1))

# Visualize SHAP values
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test_featurized[0])
