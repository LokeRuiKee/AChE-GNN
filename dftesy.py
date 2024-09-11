import pandas as pd
import torch
import deepchem as dc

# Load the dataset
df = pd.read_csv('dataset\\split\\train.csv')

# Print the DataFrame to check its structure
print("DataFrame:\n", df)

# Check the columns of the DataFrame
print("Columns in DataFrame:", df.columns)

# Ensure the 'single-class-label' column exists
if 'single-class-label' in df.columns:
    # Extract labels
    y = torch.tensor(df['single-class-label'].values, dtype=torch.long)
else:
    raise ValueError("Column 'single-class-label' not found in the DataFrame.")

