import pandas as pd
import deepchem as dc
import dgl
import torch
from graphxai.explainers import GradCAM
from graphxai.gnn_models.graph_classification import gcn
from sklearn.model_selection import KFold
import numpy as np
import csv
from datetime import datetime
from pandas import DataFrame as df

train_file = "dataset\\split\\train.csv"
test_file = "dataset\\split\\test.csv"

# Featurizer to get molecular graphs
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
loader = dc.data.CSVLoader(tasks=["single-class-label"], feature_field='SMILES', featurizer=featurizer)

# Load the entire dataset
dataset = loader.create_dataset(train_file)

# Metrics to evaluate
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score),
           dc.metrics.Metric(dc.metrics.f1_score),
           dc.metrics.Metric(dc.metrics.roc_auc_score),
           dc.metrics.Metric(dc.metrics.recall_score),
           dc.metrics.Metric(dc.metrics.matthews_corrcoef)]

# Set the best hyperparameters
best_params = {
    'learning_rate': 0.001,
    'batch_size': 50,
    'dropout': 0.3,
    'nb_epoch': 15
}

# Arrays to store results
train_scores = {metric.name: [] for metric in metrics}
test_scores = {metric.name: [] for metric in metrics}

import torch
import deepchem as dc

# Define your GCNModel (with the same architecture as the saved model)
model = dc.models.GCNModel(
    mode='classification', n_tasks=1, 
    learning_rate=best_params['learning_rate'],  # or other params
    batch_size=best_params['batch_size'],
    dropout=best_params['dropout'],
    model_dir="model"
)

# Load the existing model weights from the .pt file
model_file = "testmodel/checkpoint1.pt"
model.model.load_state_dict(torch.load(model_file))

# Set the model to evaluation mode
model.model.eval()

# Now you can evaluate or perform inference on the test set or other datasets
test_dataset = loader.create_dataset(test_file)
final_test_scores = {}
for metric in metrics:
    final_test_scores[metric.name] = model.evaluate(test_dataset, [metric])[metric.name]
    print("Final Test", metric.name, ":", final_test_scores[metric.name])

# Evaluate the model on the test dataset
test_dataset = loader.create_dataset(test_file)
final_test_scores = {}
for metric in metrics:
    final_test_scores[metric.name] = model.evaluate(test_dataset, [metric])[metric.name]
    print("Final Test", metric.name, ":", final_test_scores[metric.name])

# Save the performance evaluation metrics to a CSV file
model_name = "GCNModel"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_file = f"performance_metrics_{model_name}_{current_time}.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write headers
    writer.writerow(["Metric", "Average Train Score", "Average Test Score", "Final Test Score"])
    # Write metrics
    for metric in metrics:
        writer.writerow([metric.name, np.mean(train_scores[metric.name]), np.mean(test_scores[metric.name]), final_test_scores[metric.name]])

print(f"Performance metrics saved to {csv_file}")

# GraphXAI integration
# Assuming that each element in the dataset.X is a molecular graph with node and edge information

# Iterate through molecules in the dataset and process them one by one
for i, mol in enumerate(dataset.X):
    # Extract node and edge features from the featurizer output (MolGraphConvFeaturizer)
    atom_features = mol.get_atom_features()
    bond_features = mol.get_bond_features()

    # Construct the DGL graph
    num_atoms = len(atom_features)
    src, dst = mol.edges()
    
    graph = dgl.graph((src, dst))  # Create the graph with source and destination nodes
    graph.ndata['h'] = torch.tensor(atom_features)  # Assign node features
    graph.edata['e'] = torch.tensor(bond_features)  # Assign edge features

    # Initialize GradCAM explainer for the GCN model
    explainer = GradCAM(model=model, graph=graph, node_feats=graph.ndata['h'])

    # Example: Explain a specific node (node_idx) in the graph
    node_idx = 0  # Example index, can be changed based on your requirement
    explanation = explainer.explain_node(node_idx)

    # Output the explanation
    print(f"GradCAM Explanation for node {node_idx} in molecule {i}: {explanation}")