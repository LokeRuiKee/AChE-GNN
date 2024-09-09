import torch
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc


# Load the preprocessed data
train_file = "dataset\\split\\train.pt"
test_file = "dataset\\split\\test.pt"

train_data_list = torch.load(train_file)
test_data_list = torch.load(test_file)

import numpy as np
import deepchem as dc
from torch_geometric.data import Data

def convert_to_deepchem_dataset(data_list):
    X = []
    y = []
    for data in data_list:
        X.append(data)
        y.append(data.y.item())
    
    # Check the shapes of elements in X
    shapes = [np.array(x).shape for x in X]
    max_shape = np.max(shapes, axis=0)
    
    # Pad or truncate elements to make them uniform
    X_padded = []
    for x in X:
        x_array = np.array(x)
        padded = np.zeros(max_shape)
        slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(x_array.shape, max_shape))
        padded[slices] = x_array[slices]
        X_padded.append(padded)
    
    X_padded = np.array(X_padded)
    y = np.array(y)
    return dc.data.NumpyDataset(X_padded, y)

# Convert train and test data
train_dataset = convert_to_deepchem_dataset(train_data_list)
test_dataset = convert_to_deepchem_dataset(test_data_list)
# Initialize the model
model = dc.models.GraphConvModel(n_tasks=1, mode='classification')

# Fit the model
model.fit(train_dataset, nb_epoch=100)

# Evaluate the model
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
train_scores = model.evaluate(train_dataset, [metric])
test_scores = model.evaluate(test_dataset, [metric])

print("Train ROC-AUC Score:", train_scores['roc_auc_score'])
print("Test ROC-AUC Score:", test_scores['roc_auc_score'])