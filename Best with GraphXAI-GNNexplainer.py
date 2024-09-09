import pandas as pd
import deepchem as dc
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from graphxai.explainers import GNNExplainer
from torch_geometric.data import Data

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

# ------------------- GraphXAI Integration -------------------

# Define the torch-geometric graph data for GraphXAI
# Extract a batch of molecules to interpret
batch_data = next(iter(train_dataset.iterbatches(batch_size=1, deterministic=True)))

print(batch_data)

from torch_geometric.data import Data

# Convert DeepChem graph to PyTorch Geometric Data object (needed by GraphXAI)
def deepchem_to_torch_geometric(dc_batch):
    graphs = []
    conv_mol_array, labels, weights, ids = dc_batch
    
    for X_b, y_b in zip(conv_mol_array, labels):
        # Extract node features and edge index from the ConvMol object
        node_features = X_b.get_atom_features()  # Use the appropriate method to get node features
        edge_index = X_b.GetBonds()   # Use the appropriate method to get edge index

        # Convert to PyTorch tensor
        y_tensor = torch.tensor(y_b, dtype=torch.float)

        # Create PyTorch Geometric Data object
        graphs.append(Data(x=torch.tensor(node_features, dtype=torch.float),
                           edge_index=torch.tensor(edge_index, dtype=torch.long),
                           y=y_tensor))

    return graphs

torch_geometric_graphs = deepchem_to_torch_geometric(batch_data)

# Initialize GNNExplainer from GraphXAI
explainer = GNNExplainer(model.model, epochs=200)

# Apply explainer to one of the graph examples
for graph in torch_geometric_graphs:
    node_feat_mask, edge_mask = explainer(graph.x, graph.edge_index)

    # Visualize or analyze the explanation
    print("Node feature mask: ", node_feat_mask)
    print("Edge mask: ", edge_mask)
