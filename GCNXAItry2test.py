import pandas as pd
import deepchem as dc
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from graphxai.explainers import GradCAM
import matplotlib.pyplot as plt

# Define the GCN model
class GCN_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_3layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# Define file paths
train_file = "dataset\\split\\train.csv"
test_file = "dataset\\split\\test.csv"

# Use MolGraphConvFeaturizer to generate the appropriate features
featurizer = dc.feat.MolGraphConvFeaturizer()
loader = dc.data.CSVLoader(tasks=["single-class-label"], feature_field='SMILES', featurizer=featurizer)

# Load the entire dataset
train_dataset = loader.create_dataset(train_file)
test_dataset = loader.create_dataset(test_file)

# Initialize DeepChem model
dc_model = dc.models.GCNModel(n_tasks=1, mode='classification', batch_normalize=False)

# Train the model
dc_model.fit(train_dataset)

# Evaluate the model
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score),
           dc.metrics.Metric(dc.metrics.f1_score),
           dc.metrics.Metric(dc.metrics.roc_auc_score)]

for metric in metrics:
    print("Train", metric.name, ":", dc_model.evaluate(train_dataset, [metric]))
    print("Test", metric.name, ":", dc_model.evaluate(test_dataset, [metric]))

# Load the dataset
df = pd.read_csv('dataset\\split\\train.csv')

# Print the DataFrame to check its structure
print("DataFrame:\n", df)

# Extract labels
y = torch.tensor(df['single-class-label'].values, dtype=torch.long)

# Iterate through the dataset
for i, mol in df.iterrows():
    print(f"Mol {i}: {mol['SMILES']}")  # Inspect the SMILES string
    
    # Convert SMILES to graph using MolGraphConvFeaturizer
    graph = featurizer.featurize([mol['SMILES']])[0]  # Get the first graph

    print(dir(graph))
    print(graph)

    # Extract node features and edge indices
    atom_features = graph.node_features  # Node features
    edge_index = graph.edge_index  # Edge indices

    # Convert atom_features and edge_index to PyTorch tensors
    atom_features = torch.tensor(atom_features, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # Ensure edge_index is also a PyTorch tensor
    
    # Get the internal PyTorch model from DeepChem's GCNModel
    internal_model = dc_model.model

    # Initialize GradCAM explainer with the internal model
    explainer = GradCAM(model=internal_model)

    # Prepare inputs for the explanation method
    node_idx = 0  # Example index, can be changed based on your requirement

    # Ensure y is a tensor for the specific molecule
    y_tensor = torch.tensor([y[i]], dtype=torch.long)

    # Get the explanation for the specific node
    explanation = explainer.get_explanation_node(
        node_idx=node_idx,
        x=atom_features,
        edge_index=edge_index,
        y=y_tensor  # Pass the target labels as a tensor
    )

    # Output the explanation
    print(f"GradCAM Explanation for node {node_idx} in molecule {i}: {explanation.node_imp}")

    # Visualize the explanation
    node_importance = explanation.node_imp.detach().numpy()  # Convert to NumPy array if necessary
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(node_importance)), node_importance)
    plt.xlabel('Node Index')
    plt.ylabel('Importance Score')
    plt.title(f'GradCAM Node Importance for Node {node_idx} in Molecule {i}')
    plt.show()