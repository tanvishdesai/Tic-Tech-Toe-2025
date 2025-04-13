#!/usr/bin/env python
"""
Run script for Predictive Maintenance Solution

This script provides a command-line interface to train models
and run the dashboard for the predictive maintenance solution.
"""

import argparse
import os
import subprocess
import sys

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        import joblib
        import streamlit
        import plotly
        import tensorflow
        import scikeras
        import statsmodels
        import scipy
        print("All dependencies are installed.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def train_models(model_type="ensemble"):
    """Train the predictive maintenance models."""
    print(f"Training models with type: {model_type}")
    cmd = [sys.executable, "predictive_maintenance_model.py", "--model", model_type]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    print(stdout.decode())
    if stderr:
        print("Errors:")
        print(stderr.decode())
    
    if process.returncode == 0:
        print("Model training completed successfully.")
        return True
    else:
        print(f"Model training failed with exit code: {process.returncode}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("Starting dashboard...")
    cmd = ["streamlit", "run", "dashboard.py"]
    process = subprocess.Popen(cmd)
    print("Dashboard is running. Press Ctrl+C to stop.")
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\nDashboard stopped.")

def main():
    """Main function to parse arguments and run the appropriate commands."""
    parser = argparse.ArgumentParser(description="Run Predictive Maintenance Solution")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the models")
    train_parser.add_argument("--model", type=str, choices=["ensemble", "deep_learning"], 
                             default="ensemble", help="Type of model to train")
    
    # Dashboard parser
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the dashboard")
    
    # All-in-one parser
    all_parser = subparsers.add_parser("all", help="Train models and run dashboard")
    all_parser.add_argument("--model", type=str, choices=["ensemble", "deep_learning"], 
                           default="ensemble", help="Type of model to train")
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    if args.command == "train":
        train_models(args.model)
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "all":
        success = train_models(args.model)
        if success:
            run_dashboard()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 