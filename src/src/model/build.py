import pickle
from sklearn.linear_model import LinearRegression

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

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    os.makedirs("./model")

def build_model_and_log(config, model_name="linear_regression"):
    with wandb.init(
        project="MLOps-Diabetes-Regression", 
        name=f"Initialize Model ExecId-{args.IdExecution}", 
        job_type="initialize-model", 
        config=config) as run:
        
        config = wandb.config
        
        # Create simple linear regression model
        model = LinearRegression()
        print("Created Linear Regression model")
        
        model_artifact = wandb.Artifact(
            model_name, type="model",
            description="Simple Linear Regression model for diabetes prediction",
            metadata=dict(config))

        model_filename = f"initialized_model_{model_name}.pkl"
        model_path = f"./model/{model_filename}"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to: {model_path}")
        
        # Add to artifact
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

# Simple config
model_config = {"model_type": "LinearRegression"}

if __name__ == "__main__":
    build_model_and_log(model_config, "linear_regression")
