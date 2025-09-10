import numpy as np
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
# Testing
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Load the diabetes dataset from sklearn and split into train/val/test
    """
    
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {diabetes.feature_names}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create datasets as dictionaries
    datasets = {
        'training': {'X': X_train, 'y': y_train},
        'validation': {'X': X_val, 'y': y_val},
        'test': {'X': X_test, 'y': y_test}
    }
    
    return datasets, diabetes.feature_names

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps-Diabetes-Regression",
        name=f"Load Raw Data ExecId-{args.IdExecution}", 
        job_type="load-data") as run:
        
        datasets, feature_names = load()  # separate code for loading the datasets

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "diabetes-raw", type="dataset",
            description="raw Diabetes dataset from sklearn, split into train/val/test",
            metadata={
                "source": "sklearn.datasets.load_diabetes",
                "sizes": {name: data['X'].shape[0] for name, data in datasets.items()},
                "features": len(feature_names),
                "feature_names": feature_names.tolist(),
                "target": "diabetes_progression",
                "target_type": "continuous"
            })

        for name, data in datasets.items():
            # 🐣 Store numpy arrays as pickle files
            with raw_data.new_file(name + ".pkl", mode="wb") as file:
                pickle.dump(data, file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing workflow
if __name__ == "__main__":
    load_and_log()
