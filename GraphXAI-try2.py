import pandas as pd
import deepchem as dc
import dgl
import graphxai
from graphxai.explainers import GradCAM
from graphxai.gnn_models.graph_classification import gcn
from sklearn.model_selection import KFold
import numpy as np
import csv
from datetime import datetime

train_file = "dataset\\split\\train.csv"
test_file = "dataset\\split\\test.csv"

featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
loader = dc.data.CSVLoader(tasks=["single-class-label"], feature_field='SMILES', featurizer=featurizer)

# Load the entire dataset
dataset = loader.create_dataset(train_file)

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrics to evaluate
metrics = [dc.metrics.Metric(dc.metrics.accuracy_score),
           dc.metrics.Metric(dc.metrics.f1_score),
           dc.metrics.Metric(dc.metrics.roc_auc_score),
           dc.metrics.Metric(dc.metrics.recall_score),
           dc.metrics.Metric(dc.metrics.matthews_corrcoef)]

# Arrays to store results
train_scores = {metric.name: [] for metric in metrics}
test_scores = {metric.name: [] for metric in metrics}

# Perform 5-fold cross-validation
for train_index, val_index in kf.split(dataset.ids):
    train_dataset = dataset.select(train_index)
    val_dataset = dataset.select(val_index)

    # Initialize the model
    model = dc.models.GCNModel(mode='classification', n_tasks=1, batch_size=50, learning_rate=0.001, use_queue=False, model_dir="model")

    # Train the model
    model.fit(train_dataset, nb_epoch=15)

    # Evaluate the model on training and validation sets
    for metric in metrics:
        train_score = model.evaluate(train_dataset, [metric])[metric.name]
        val_score = model.evaluate(val_dataset, [metric])[metric.name]
        train_scores[metric.name].append(train_score)
        test_scores[metric.name].append(val_score)

# Print average scores for each metric
for metric in metrics:
    print(f"Average Train {metric.name}: {np.mean(train_scores[metric.name])}")
    print(f"Average Test {metric.name}: {np.mean(test_scores[metric.name])}")

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