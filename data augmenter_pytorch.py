import pandas as pd
from sklearn.model_selection import train_test_split
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles

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

# Convert augmented SMILES to molecular graphs
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return from_smiles(smiles)

train_graphs = [smiles_to_graph(smiles) for smiles in augmented_smiles]
train_labels = torch.tensor(augmented_labels, dtype=torch.long)

# Filter out None values
train_graphs = [graph for graph in train_graphs if graph is not None]

# Create PyTorch Geometric Data objects
train_data_list = [Data(x=graph.x, edge_index=graph.edge_index, y=label) for graph, label in zip(train_graphs, train_labels)]

# Repeat the process for test data without augmentation
test_graphs = [smiles_to_graph(smiles) for smiles in X_test]
test_labels = torch.tensor(y_test.values, dtype=torch.long)
test_graphs = [graph for graph in test_graphs if graph is not None]
test_data_list = [Data(x=graph.x, edge_index=graph.edge_index, y=label) for graph, label in zip(test_graphs, test_labels)]

# Save the processed data
base_dir = "dataset\\split"
train_file = os.path.join(base_dir, "train.pt")
test_file = os.path.join(base_dir, "test.pt")

# Create the directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

torch.save(train_data_list, train_file)
torch.save(test_data_list, test_file)