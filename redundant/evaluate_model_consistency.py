"""
Script to evaluate the consistency of the predictive maintenance model
over multiple runs with different random seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_model_inference import generate_synthetic_test_data, test_model_inference
import os
import time

# Create output directory if it doesn't exist
os.makedirs('test_outputs/consistency', exist_ok=True)

def run_multiple_evaluations(machine_type, num_runs=5, samples_per_run=200):
    """
    Run multiple evaluations of the model with different random seeds.
    
    Args:
        machine_type: The type of machine to test
        num_runs: Number of evaluation runs to perform
        samples_per_run: Number of samples to generate per run
        
    Returns:
        DataFrame with evaluation metrics for each run
    """
    print(f"\nRunning {num_runs} evaluations for {machine_type}...")
    
    # Initialize results tracking
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}:")
        
        # Set a different random seed for each run
        np.random.seed(42 + run)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run test and get results
            eval_result = test_model_inference(machine_type)
            
            if eval_result:
                # Record time taken
                duration = time.time() - start_time
                
                # Add results to list
                results.append({
                    'run': run + 1,
                    'accuracy': eval_result['accuracy'],
                    'roc_auc': eval_result['roc_auc'],
                    'pr_auc': eval_result['pr_auc'],
                    'duration': duration
                })
                
                # Rename the prediction results file to include the run number
                os.rename(
                    f'test_outputs/{machine_type}_prediction_results.csv',
                    f'test_outputs/consistency/{machine_type}_run{run+1}_prediction_results.csv'
                )
                
        except Exception as e:
            print(f"Error in run {run+1}: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        'mean': results_df.mean(),
        'std': results_df.std(),
        'min': results_df.min(),
        'max': results_df.max()
    }
    
    # Print summary
    print("\nPerformance Summary:")
    print(f"Average Accuracy: {stats['mean']['accuracy']:.4f} ± {stats['std']['accuracy']:.4f}")
    print(f"Average ROC-AUC: {stats['mean']['roc_auc']:.4f} ± {stats['std']['roc_auc']:.4f}")
    print(f"Average PR-AUC: {stats['mean']['pr_auc']:.4f} ± {stats['std']['pr_auc']:.4f}")
    print(f"Average Duration: {stats['mean']['duration']:.2f} seconds")
    
    # Plot consistency metrics
    plt.figure(figsize=(12, 8))
    
    metrics = ['accuracy', 'roc_auc', 'pr_auc']
    colors = ['blue', 'green', 'red']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.bar(results_df['run'], results_df[metric], color=colors[i])
        plt.axhline(y=stats['mean'][metric], color='black', linestyle='--', label='Mean')
        plt.fill_between(
            [0.5, num_runs + 0.5], 
            stats['mean'][metric] - stats['std'][metric],
            stats['mean'][metric] + stats['std'][metric],
            color='gray', alpha=0.2, label='±1 Std Dev'
        )
        plt.title(f"{metric} across {num_runs} runs")
        plt.xlabel("Run")
        plt.ylabel(metric)
        plt.ylim([0, 1.05])  # Metrics range from 0 to 1
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot run duration
    plt.subplot(2, 2, 4)
    plt.bar(results_df['run'], results_df['duration'], color='purple')
    plt.axhline(y=stats['mean']['duration'], color='black', linestyle='--', label='Mean')
    plt.title(f"Run Duration across {num_runs} runs")
    plt.xlabel("Run")
    plt.ylabel("Duration (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'test_outputs/consistency/{machine_type}_consistency_metrics.png')
    
    # Save results to CSV
    results_df.to_csv(f'test_outputs/consistency/{machine_type}_consistency_results.csv', index=False)
    
    return results_df, stats

if __name__ == "__main__":
    # Define machine types to evaluate
    machine_types = [
        "siemens_motor",
        # Uncomment to test other machine types
        # "abb_bearing",
        # "haas_cnc",
        # "grundfos_pump",
        # "carrier_chiller"
    ]
    
    # Number of evaluation runs
    num_runs = 3  # Use a small number for quick testing
    
    all_results = {}
    all_stats = {}
    
    for machine_type in machine_types:
        try:
            results_df, stats = run_multiple_evaluations(
                machine_type=machine_type,
                num_runs=num_runs
            )
            all_results[machine_type] = results_df
            all_stats[machine_type] = stats
        except Exception as e:
            print(f"Error evaluating {machine_type}: {str(e)}")
    
    # Print overall summary
    print("\nOverall Summary:")
    for machine_type, stats in all_stats.items():
        print(f"{machine_type}: Avg Accuracy = {stats['mean']['accuracy']:.4f} ± {stats['std']['accuracy']:.4f}, " +
              f"Avg PR-AUC = {stats['mean']['pr_auc']:.4f} ± {stats['std']['pr_auc']:.4f}") 