import pandas as pd
import deepchem as dc
import dgl
import graphxai
from graphxai.explainers import GradCAM
from graphxai.gnn_models.graph_classification import gcn
from sklearn.model_selection import KFold, ParameterGrid
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

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01], # 1st best model 0.0001, 0.9, 600/64 is best
    'batch_size': [32, 64, 128], # 1st best model 64 is best
    'dropout': [0.2, 0.3, 0.5],
    'nb_epoch': [5, 10, 15, 30]
}

# Initialize arrays to store the best results
best_params = None
best_score = -np.inf
results = []

# Perform grid search for each combination of hyperparameters
param_combinations = ParameterGrid(param_grid)

for params in param_combinations:
    print(f"Training with parameters: {params}")

    # Initialize arrays to store results for each fold
    train_scores = {metric.name: [] for metric in metrics}
    test_scores = {metric.name: [] for metric in metrics}

    # Perform 5-fold cross-validation
    for train_index, val_index in kf.split(dataset.ids):
        train_dataset = dataset.select(train_index)
        val_dataset = dataset.select(val_index)

        # Initialize the model with current parameter combination
        model = dc.models.GCNModel(
            mode='classification', n_tasks=1, 
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            dropout=params['dropout'],
            model_dir="model"
        )

        # Train the model
        model.fit(train_dataset, nb_epoch=params['nb_epoch'])

        # Evaluate the model on training and validation sets
        for metric in metrics:
            train_score = model.evaluate(train_dataset, [metric])[metric.name]
            val_score = model.evaluate(val_dataset, [metric])[metric.name]
            train_scores[metric.name].append(train_score)
            test_scores[metric.name].append(val_score)

    # Calculate average validation score (e.g., using accuracy or f1_score)
    mean_val_score = np.mean([np.mean(test_scores['accuracy_score']), np.mean(test_scores['f1_score'])])

    # Save the best combination of hyperparameters
    if mean_val_score > best_score:
        best_score = mean_val_score
        best_params = params

    # Store results for this parameter set
    results.append({
        'params': params,
        'val_score': mean_val_score
    })

# Print the best parameters and their validation score
print("Best Parameters:", best_params)
print("Best Validation Score:", best_score)

# Retrain the best model using the entire dataset
best_model = dc.models.GCNModel(
    mode='classification', n_tasks=1, 
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    dropout=best_params['dropout']
)

best_model.fit(dataset, nb_epoch=best_params['nb_epoch'])

# Evaluate the best model on the test dataset
test_dataset = loader.create_dataset(test_file)
final_test_scores = {}
for metric in metrics:
    final_test_scores[metric.name] = best_model.evaluate(test_dataset, [metric])[metric.name]
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

# Save the grid search results to a CSV file
grid_search_csv = f"grid_search_results_{model_name}_{current_time}.csv"
with open(grid_search_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write headers
    writer.writerow(["Learning Rate", "Batch Size", "Dropout", "Epochs", "Validation Score"])
    # Write results for each parameter set
    for result in results:
        writer.writerow([result['params']['learning_rate'],
                         result['params']['batch_size'],
                         result['params']['dropout'],
                         result['params']['nb_epoch'],
                         result['val_score']])

print(f"Grid search results saved to {grid_search_csv}")
