#!/usr/bin/env python3
"""
RAMSeS Full Testbed Runner
Runs the RAMSeS framework across all datasets and generates aggregated results.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess
import traceback
from Utils.paths import results_dir

class TestbedRunner:
    def __init__(self, config_file: str, output_dir: str = None):
        self.config_file = config_file
        self.output_dir = output_dir or results_dir("testbed_aggregated")
        self.results = []
        self.start_time = None
        self.end_time = None
        
        # Dataset configurations
        self.datasets = [
            # SKAB datasets (16 entities)
            ("SKAB", "0"), ("SKAB", "1"), ("SKAB", "2"), ("SKAB", "3"),
            ("SKAB", "4"), ("SKAB", "5"), ("SKAB", "6"), ("SKAB", "7"),
            ("SKAB", "8"), ("SKAB", "9"), ("SKAB", "10"), ("SKAB", "11"),
            ("SKAB", "12"), ("SKAB", "13"), ("SKAB", "14"), ("SKAB", "15"),
            
            # SKAB_valve1 datasets (16 entities)
            ("SKAB_valve1", "0"), ("SKAB_valve1", "1"), ("SKAB_valve1", "2"),
            ("SKAB_valve1", "3"), ("SKAB_valve1", "4"), ("SKAB_valve1", "5"),
            ("SKAB_valve1", "6"), ("SKAB_valve1", "7"), ("SKAB_valve1", "8"),
            ("SKAB_valve1", "9"), ("SKAB_valve1", "10"), ("SKAB_valve1", "11"),
            ("SKAB_valve1", "12"), ("SKAB_valve1", "13"), ("SKAB_valve1", "14"),
            ("SKAB_valve1", "15"),
            
            # SKAB_valve2 datasets (4 entities)
            ("SKAB_valve2", "0"), ("SKAB_valve2", "1"),
            ("SKAB_valve2", "2"), ("SKAB_valve2", "3"),
        ]
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_single_experiment(self, dataset: str, entity: str) -> Dict:
        """Run RAMSeS on a single dataset/entity combination."""
        print(f"\n{'='*80}")
        print(f"Running: {dataset}/{entity}")
        print(f"{'='*80}")
        
        experiment_start = time.time()
        
        try:
            # Run the app.py script
            cmd = [
                sys.executable, "app.py",
                "-c", self.config_file,
                "--dataset", dataset,
                "--entity", entity
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            experiment_end = time.time()
            runtime = experiment_end - experiment_start
            
            if result.returncode != 0:
                print(f"❌ FAILED: {dataset}/{entity}")
                print(f"Error: {result.stderr[-500:]}")  # Last 500 chars
                return {
                    "dataset": dataset,
                    "entity": entity,
                    "status": "failed",
                    "error": result.stderr[-500:],
                    "runtime": runtime
                }
            
            # Load the JSON results
            json_pattern = results_dir("comprehensive", dataset, entity) + "results_*.json"
            json_files = list(Path(".").glob(json_pattern))
            
            if not json_files:
                print(f"⚠️  WARNING: No results file found for {dataset}/{entity}")
                return {
                    "dataset": dataset,
                    "entity": entity,
                    "status": "no_results",
                    "runtime": runtime
                }
            
            # Get the most recent results file
            latest_json = sorted(json_files)[-1]
            
            with open(latest_json, 'r') as f:
                results_data = json.load(f)
            
            print(f"✅ SUCCESS: {dataset}/{entity} (Runtime: {runtime:.2f}s)")
            
            # Extract metrics
            return {
                "dataset": dataset,
                "entity": entity,
                "status": "success",
                "runtime": runtime,
                "ga_f1": results_data['results']['ga'].get('f1', 0.0),
                "ga_pr_auc": results_data['results']['ga'].get('pr_auc', 0.0),
                "thompson_f1": results_data['results']['thompson'].get('f1', 0.0),
                "thompson_pr_auc": results_data['results']['thompson'].get('pr_auc', 0.0),
                "gan_f1": results_data['results']['gan'].get('f1', 0.0),
                "gan_pr_auc": results_data['results']['gan'].get('pr_auc', 0.0),
                "borderline_f1": results_data['results']['borderline'].get('f1', 0.0),
                "borderline_pr_auc": results_data['results']['borderline'].get('pr_auc', 0.0),
                "montecarlo_f1": results_data['results']['montecarlo'].get('f1', 0.0),
                "montecarlo_pr_auc": results_data['results']['montecarlo'].get('pr_auc', 0.0),
                "ga_duration": results_data['timings'].get('ga_duration', 0.0),
                "thompson_duration": results_data['timings'].get('thompson_duration', 0.0),
                "gan_duration": results_data['timings'].get('gan_duration', 0.0),
                "borderline_duration": results_data['timings'].get('borderline_duration', 0.0),
                "montecarlo_duration": results_data['timings'].get('montecarlo_duration', 0.0),
                "total_runtime": results_data['metadata'].get('total_runtime_seconds', 0.0),
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏱️  TIMEOUT: {dataset}/{entity}")
            return {
                "dataset": dataset,
                "entity": entity,
                "status": "timeout",
                "runtime": 3600.0
            }
        except Exception as e:
            print(f"❌ ERROR: {dataset}/{entity}")
            print(f"Exception: {str(e)}")
            traceback.print_exc()
            return {
                "dataset": dataset,
                "entity": entity,
                "status": "error",
                "error": str(e),
                "runtime": time.time() - experiment_start
            }
    
    def run_all_experiments(self):
        """Run RAMSeS on all datasets."""
        self.start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Starting Full Testbed Run")
        print(f"Total datasets: {len(self.datasets)}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        for i, (dataset, entity) in enumerate(self.datasets, 1):
            print(f"\nProgress: {i}/{len(self.datasets)} ({i/len(self.datasets)*100:.1f}%)")
            result = self.run_single_experiment(dataset, entity)
            self.results.append(result)
            
            # Save intermediate results after each experiment
            self.save_intermediate_results()
        
        self.end_time = time.time()
        print(f"\n{'='*80}")
        print(f"Testbed Run Completed")
        print(f"Total time: {(self.end_time - self.start_time)/3600:.2f} hours")
        print(f"{'='*80}\n")
    
    def save_intermediate_results(self):
        """Save intermediate results (called after each experiment)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(self.output_dir, f"intermediate_results_{timestamp}.json")
        
        with open(intermediate_file, 'w') as f:
            json.dump({
                "experiments": self.results,
                "timestamp": timestamp,
                "completed": len(self.results),
                "total": len(self.datasets)
            }, f, indent=2)
    
    def generate_aggregated_report(self):
        """Generate comprehensive aggregated report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter successful results
        df_success = df[df['status'] == 'success'].copy()
        
        if len(df_success) == 0:
            print("⚠️  No successful experiments to aggregate!")
            return
        
        # Calculate statistics
        print(f"\nGenerating aggregated report...")
        print(f"Successful experiments: {len(df_success)}/{len(df)}")
        
        # 1. Save detailed CSV with all results
        csv_file = os.path.join(self.output_dir, f"detailed_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"✅ Detailed CSV saved: {csv_file}")
        
        # 2. Generate summary statistics
        summary_file = os.path.join(self.output_dir, f"summary_statistics_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RAMSeS TESTBED AGGREGATED RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Successful: {len(df_success)}\n")
            f.write(f"Failed: {len(df[df['status'] == 'failed'])}\n")
            f.write(f"Timeout: {len(df[df['status'] == 'timeout'])}\n")
            f.write(f"Error: {len(df[df['status'] == 'error'])}\n\n")
            
            # Overall statistics
            f.write("="*80 + "\n")
            f.write("OVERALL PERFORMANCE METRICS (Mean ± Std)\n")
            f.write("="*80 + "\n\n")
            
            metrics = [
                ('ga_f1', 'Genetic Algorithm F1'),
                ('ga_pr_auc', 'Genetic Algorithm PR-AUC'),
                ('thompson_f1', 'Thompson Sampling F1'),
                ('thompson_pr_auc', 'Thompson Sampling PR-AUC'),
                ('gan_f1', 'GAN Robustness F1'),
                ('gan_pr_auc', 'GAN Robustness PR-AUC'),
                ('borderline_f1', 'Borderline Sensitivity F1'),
                ('borderline_pr_auc', 'Borderline Sensitivity PR-AUC'),
                ('montecarlo_f1', 'Monte Carlo F1'),
                ('montecarlo_pr_auc', 'Monte Carlo PR-AUC'),
            ]
            
            for metric, name in metrics:
                if metric in df_success.columns:
                    mean = df_success[metric].mean()
                    std = df_success[metric].std()
                    f.write(f"{name:40s}: {mean:.6f} ± {std:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("COMPUTATIONAL OVERHEAD (seconds)\n")
            f.write("="*80 + "\n\n")
            
            timing_metrics = [
                ('ga_duration', 'Genetic Algorithm'),
                ('thompson_duration', 'Thompson Sampling'),
                ('gan_duration', 'GAN Robustness'),
                ('borderline_duration', 'Borderline Sensitivity'),
                ('montecarlo_duration', 'Monte Carlo'),
                ('total_runtime', 'Total Pipeline'),
            ]
            
            for metric, name in timing_metrics:
                if metric in df_success.columns:
                    mean = df_success[metric].mean()
                    std = df_success[metric].std()
                    total = df_success[metric].sum()
                    f.write(f"{name:30s}: {mean:8.2f}s ± {std:8.2f}s  (Total: {total:10.2f}s)\n")
            
            # Per-dataset statistics
            f.write("\n" + "="*80 + "\n")
            f.write("PER-DATASET STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            for dataset in df_success['dataset'].unique():
                df_dataset = df_success[df_success['dataset'] == dataset]
                f.write(f"\n{dataset} ({len(df_dataset)} entities):\n")
                f.write(f"  GA F1:         {df_dataset['ga_f1'].mean():.6f} ± {df_dataset['ga_f1'].std():.6f}\n")
                f.write(f"  Thompson F1:   {df_dataset['thompson_f1'].mean():.6f} ± {df_dataset['thompson_f1'].std():.6f}\n")
                f.write(f"  GAN F1:        {df_dataset['gan_f1'].mean():.6f} ± {df_dataset['gan_f1'].std():.6f}\n")
                f.write(f"  Borderline F1: {df_dataset['borderline_f1'].mean():.6f} ± {df_dataset['borderline_f1'].std():.6f}\n")
                f.write(f"  Monte Carlo F1:{df_dataset['montecarlo_f1'].mean():.6f} ± {df_dataset['montecarlo_f1'].std():.6f}\n")
                f.write(f"  Avg Runtime:   {df_dataset['total_runtime'].mean():.2f}s ± {df_dataset['total_runtime'].std():.2f}s\n")
        
        print(f"✅ Summary statistics saved: {summary_file}")
        
        # 3. Generate JSON aggregate
        json_file = os.path.join(self.output_dir, f"aggregated_results_{timestamp}.json")
        
        aggregate_data = {
            "metadata": {
                "timestamp": timestamp,
                "total_experiments": len(df),
                "successful_experiments": len(df_success),
                "total_runtime_hours": (self.end_time - self.start_time) / 3600 if self.end_time else 0,
            },
            "overall_metrics": {
                metric: {
                    "mean": float(df_success[metric].mean()) if metric in df_success.columns else 0.0,
                    "std": float(df_success[metric].std()) if metric in df_success.columns else 0.0,
                    "min": float(df_success[metric].min()) if metric in df_success.columns else 0.0,
                    "max": float(df_success[metric].max()) if metric in df_success.columns else 0.0,
                }
                for metric, _ in metrics
            },
            "timing_metrics": {
                metric: {
                    "mean": float(df_success[metric].mean()) if metric in df_success.columns else 0.0,
                    "std": float(df_success[metric].std()) if metric in df_success.columns else 0.0,
                    "total": float(df_success[metric].sum()) if metric in df_success.columns else 0.0,
                }
                for metric, _ in timing_metrics
            },
            "per_dataset_metrics": {
                dataset: {
                    "count": int(len(df_dataset)),
                    "ga_f1_mean": float(df_dataset['ga_f1'].mean()),
                    "thompson_f1_mean": float(df_dataset['thompson_f1'].mean()),
                    "gan_f1_mean": float(df_dataset['gan_f1'].mean()),
                    "borderline_f1_mean": float(df_dataset['borderline_f1'].mean()),
                    "montecarlo_f1_mean": float(df_dataset['montecarlo_f1'].mean()),
                    "avg_runtime": float(df_dataset['total_runtime'].mean()),
                }
                for dataset in df_success['dataset'].unique()
                for df_dataset in [df_success[df_success['dataset'] == dataset]]
            },
            "detailed_results": self.results
        }
        
        with open(json_file, 'w') as f:
            json.dump(aggregate_data, f, indent=2)
        
        print(f"✅ JSON aggregate saved: {json_file}")
        
        # 4. Generate comparison table
        comparison_file = os.path.join(self.output_dir, f"method_comparison_{timestamp}.csv")
        
        comparison_data = []
        for method in ['ga', 'thompson', 'gan', 'borderline', 'montecarlo']:
            f1_col = f'{method}_f1'
            pr_col = f'{method}_pr_auc'
            time_col = f'{method}_duration'
            
            if f1_col in df_success.columns:
                comparison_data.append({
                    'Method': method.upper().replace('_', ' '),
                    'F1_Mean': df_success[f1_col].mean(),
                    'F1_Std': df_success[f1_col].std(),
                    'PR_AUC_Mean': df_success[pr_col].mean(),
                    'PR_AUC_Std': df_success[pr_col].std(),
                    'Time_Mean_s': df_success[time_col].mean(),
                    'Time_Std_s': df_success[time_col].std(),
                    'Time_Total_s': df_success[time_col].sum(),
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(comparison_file, index=False)
        print(f"✅ Method comparison saved: {comparison_file}")
        
        print(f"\n{'='*80}")
        print(f"All reports generated in: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run RAMSeS on full testbed')
    parser.add_argument('-c', '--config', default='Configs/custom_config.yml',
                        help='Path to config file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory for aggregated results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last completed experiment')
    
    args = parser.parse_args()
    
    runner = TestbedRunner(args.config, args.output)
    runner.run_all_experiments()
    runner.generate_aggregated_report()
    
    print("\n✅ Full testbed run completed successfully!")


if __name__ == "__main__":
    main()
