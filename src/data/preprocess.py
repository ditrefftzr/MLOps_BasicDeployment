import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

#testing workflow
import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

def preprocess(X_train, X_val, X_test, y_train, y_val, y_test, scaler_type="standard", normalize_target=False):
    """
    Preprocess the diabetes dataset
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target vectors
        scaler_type: "standard", "minmax", or "none"
        normalize_target: Whether to normalize target values
    """
    
    print(f"Original data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Target ranges - Train: [{y_train.min():.1f}, {y_train.max():.1f}]")
    
    # Initialize scaler based on type
    if scaler_type == "standard":
        feature_scaler = StandardScaler()
    elif scaler_type == "minmax":
        feature_scaler = MinMaxScaler()
    else:
        feature_scaler = None
    
    # Scale features if requested
    if feature_scaler is not None:
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        X_test_scaled = feature_scaler.transform(X_test)
        
        print(f"Applied {scaler_type} scaling to features")
        print(f"  Train feature range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    else:
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        print("No feature scaling applied")
    
    # Scale target if requested
    target_scaler = None
    if normalize_target:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"Applied target scaling")
        print(f"  Train target range: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
    else:
        y_train_scaled = y_train.copy()
        y_val_scaled = y_val.copy()
        y_test_scaled = y_test.copy()
        print("No target scaling applied")
    
    # Create processed datasets
    processed_datasets = {
        'training': {'X': X_train_scaled, 'y': y_train_scaled},
        'validation': {'X': X_val_scaled, 'y': y_val_scaled},
        'test': {'X': X_test_scaled, 'y': y_test_scaled}
    }
    
    # Store scalers for later use
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    return processed_datasets, scalers

def preprocess_and_log(steps):
    with wandb.init(
        project="MLOps-Diabetes-Regression",
        name=f"Preprocess Data ExecId-{args.IdExecution}", 
        job_type="preprocess-data") as run:    
        
        processed_data = wandb.Artifact(
            "diabetes-preprocess", type="dataset",
            description="Preprocessed Diabetes dataset",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('diabetes-raw:latest')

        # 📥 if need be, download the artifact
        raw_dataset_path = raw_data_artifact.download(root="./data/artifacts/")
        
        # Load raw datasets
        raw_datasets = {}
        for split in ["training", "validation", "test"]:
            raw_datasets[split] = read(raw_dataset_path, split)
        
        # Extract X, y for preprocessing
        X_train, y_train = raw_datasets['training']['X'], raw_datasets['training']['y']
        X_val, y_val = raw_datasets['validation']['X'], raw_datasets['validation']['y']
        X_test, y_test = raw_datasets['test']['X'], raw_datasets['test']['y']
        
        # Apply preprocessing
        processed_datasets, scalers = preprocess(
            X_train, X_val, X_test, y_train, y_val, y_test, **steps
        )
        
        # Save processed datasets
        for name, data in processed_datasets.items():
            with processed_data.new_file(name + ".pkl", mode="wb") as file:
                pickle.dump(data, file)
        
        # Save scalers for later use in training/inference
        with processed_data.new_file("scalers.pkl", mode="wb") as file:
            pickle.dump(scalers, file)

        run.log_artifact(processed_data)

def read(data_dir, split):
    """Read pickled data from directory"""
    filename = split + ".pkl"
    with open(os.path.join(data_dir, filename), 'rb') as file:
        data = pickle.load(file)
    return data

# Preprocessing configuration
steps = {
    "scaler_type": "standard",  # "standard", "minmax", or "none"
    "normalize_target": False   # Whether to scale target values
}

if __name__ == "__main__":
    preprocess_and_log(steps)
