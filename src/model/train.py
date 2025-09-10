import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def read_data(data_dir, split):
    """Read processed data from directory"""
    filename = split + ".pkl"
    with open(os.path.join(data_dir, filename), 'rb') as file:
        data = pickle.load(file)
    return data['X'], data['y']

def evaluate_model(model, X, y, split_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"{split_name} Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²:  {r2:.4f}")
    
    return {
        f"{split_name}/mse": mse,
        f"{split_name}/mae": mae,
        f"{split_name}/r2": r2
    }

def train_and_log():
    with wandb.init(
        project="MLOps-Diabetes-Regression", 
        name=f"Train Model ExecId-{args.IdExecution}", 
        job_type="train-model") as run:
        
        # Download preprocessed data
        print("Downloading preprocessed data...")
        data_artifact = run.use_artifact('diabetes-preprocess:latest')
        data_dir = data_artifact.download()
        
        # Load training and validation data
        X_train, y_train = read_data(data_dir, "training")
        X_val, y_val = read_data(data_dir, "validation")
        X_test, y_test = read_data(data_dir, "test")
        
        print(f"Training data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")
        print(f"Test data: {X_test.shape}")
        
        # Download model
        print("Downloading initialized model...")
        model_artifact = run.use_artifact("linear_regression:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model_linear_regression.pkl")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("Model loaded successfully")
        
        # Train the model
        print("Training model...")
        model.fit(X_train, y_train)
        print("Training completed!")
        
        # Evaluate on all splits
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        val_metrics = evaluate_model(model, X_val, y_val, "validation")
        test_metrics = evaluate_model(model, X_test, y_test, "test")
        
        # Log all metrics to wandb
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        wandb.log(all_metrics)
        
        # Log summary metrics
        run.summary.update(all_metrics)
        
        # Save trained model
        print("Saving trained model...")
        trained_model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained Linear Regression model for diabetes prediction",
            metadata={
                "model_type": "LinearRegression",
                "training_samples": len(X_train),
                "features": X_train.shape[1]
            })
        
        # Save locally first push!
        trained_model_path = "trained_model.pkl"
        with open(trained_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Add to artifact
        trained_model_artifact.add_file(trained_model_path)
        
        # Log artifact
        run.log_artifact(trained_model_artifact)
        
        print("Trained model artifact saved to WandB")
        print(f"Final validation R²: {val_metrics['validation/r2']:.4f}")

if __name__ == "__main__":
    train_and_log()
