#!/usr/bin/env python3
"""
Comprehensive Testbed Runner for RAMSeS Framework

This script runs the RAMSeS framework over multiple datasets and collects:
- Overall computational overhead
- Average computational overhead
- Average F1 scores (per module and overall)
- Average memory footprint

Organizes results by dataset domain (SKAB, SMD, etc.)
"""

import os
import sys
import subprocess
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
from datetime import datetime
import re
from collections import defaultdict
import logging
from typing import Dict, List, Tuple
from Utils.paths import results_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.measurements = []
    
    def update(self):
        """Record current memory usage"""
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.measurements.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb
    
    def get_average(self):
        """Get average memory usage"""
        return np.mean(self.measurements) if self.measurements else 0
    
    def get_peak(self):
        """Get peak memory usage"""
        return self.peak_memory
    
    def reset(self):
        """Reset measurements"""
        self.measurements = []
        self.peak_memory = 0


class TestbedRunner:
    """Run RAMSeS testbed across multiple datasets"""
    
    def __init__(self, dataset_list_file: str, output_base_dir: str = "testbed_results", 
                 timeout: int = 3600, parallel: str = 'false'):
        """
        Initialize testbed runner
        
        Parameters
        ----------
        dataset_list_file : str
            Path to CSV file with columns: file_name, domain_name
        output_base_dir : str
            Base directory for storing results
        timeout : int
            Timeout in seconds for each dataset execution (default: 3600 = 1 hour)
        parallel : str
            Whether to use parallel model selection (default: 'false')
        """
        self.dataset_list_file = dataset_list_file
        self.output_base_dir = output_base_dir
        self.timeout = timeout
        self.parallel = parallel
        self.results_by_domain = defaultdict(list)
        self.memory_monitor = MemoryMonitor()
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Load dataset list
        self.datasets_df = pd.read_csv(dataset_list_file)
        logger.info(f"Loaded {len(self.datasets_df)} datasets from {dataset_list_file}")
        
        # Group by domain
        self.domains = self.datasets_df['domain_name'].unique()
        logger.info(f"Found {len(self.domains)} domains: {list(self.domains)}")
    
    def parse_comprehensive_results(self, results_file: str) -> Dict:
        """
        Parse comprehensive results file to extract metrics
        
        Parameters
        ----------
        results_file : str
            Path to comprehensive results file
            
        Returns
        -------
        Dict with extracted metrics
        """
        if not os.path.exists(results_file):
            logger.warning(f"Results file not found: {results_file}")
            return {}
        
        metrics = {
            'timing': {},
            'ga': {},
            'thompson': {},
            'gan': {},
            'borderline': {},
            'monte_carlo': {},
            'final_decision': {}
        }
        
        with open(results_file, 'r') as f:
            content = f.read()
        
        # Extract timing information
        timing_pattern = r'(\d+_\w+)\s*:\s*([\d.]+)s'
        for match in re.finditer(timing_pattern, content):
            module_name = match.group(1)
            time_val = float(match.group(2))
            metrics['timing'][module_name] = time_val
        
        # Extract end-to-end time
        e2e_match = re.search(r'End-to-End Time\s*:\s*([\d.]+)s', content)
        if e2e_match:
            metrics['timing']['end_to_end'] = float(e2e_match.group(1))
        
        # Extract GA metrics
        ga_f1_match = re.search(r'GENETIC ALGORITHM.*?F1 Score\s*:\s*([\d.]+)', content, re.DOTALL)
        if ga_f1_match:
            metrics['ga']['f1'] = float(ga_f1_match.group(1))
        
        ga_prauc_match = re.search(r'GENETIC ALGORITHM.*?PR-AUC\s*:\s*([\d.]+)', content, re.DOTALL)
        if ga_prauc_match:
            metrics['ga']['pr_auc'] = float(ga_prauc_match.group(1))
        
        # Extract Thompson Sampling metrics
        thompson_f1_match = re.search(r'THOMPSON SAMPLING.*?F1 Score\s*:\s*([-\d.]+)', content, re.DOTALL)
        if thompson_f1_match:
            metrics['thompson']['f1'] = float(thompson_f1_match.group(1))
        
        thompson_prauc_match = re.search(r'THOMPSON SAMPLING.*?PR-AUC\s*:\s*([\d.]+)', content, re.DOTALL)
        if thompson_prauc_match:
            metrics['thompson']['pr_auc'] = float(thompson_prauc_match.group(1))
        
        # Extract GAN Robustness metrics
        gan_f1_match = re.search(r'GAN Robustness Test:.*?F1 Score\s*:\s*([\d.]+)', content, re.DOTALL)
        if gan_f1_match:
            metrics['gan']['f1'] = float(gan_f1_match.group(1))
        
        gan_prauc_match = re.search(r'GAN Robustness Test:.*?PR-AUC\s*:\s*([\d.]+)', content, re.DOTALL)
        if gan_prauc_match:
            metrics['gan']['pr_auc'] = float(gan_prauc_match.group(1))
        
        # Extract Borderline metrics
        borderline_f1_match = re.search(r'Borderline Sensitivity Test:.*?F1 Score\s*:\s*([\d.]+)', content, re.DOTALL)
        if borderline_f1_match:
            metrics['borderline']['f1'] = float(borderline_f1_match.group(1))
        
        borderline_prauc_match = re.search(r'Borderline Sensitivity Test:.*?PR-AUC\s*:\s*([\d.]+)', content, re.DOTALL)
        if borderline_prauc_match:
            metrics['borderline']['pr_auc'] = float(borderline_prauc_match.group(1))
        
        # Extract Monte Carlo metrics
        mc_f1_match = re.search(r'Monte Carlo Simulation:.*?Best F1 Score\s*:\s*([\d.]+)', content, re.DOTALL)
        if mc_f1_match:
            metrics['monte_carlo']['f1'] = float(mc_f1_match.group(1))
        
        mc_prauc_match = re.search(r'Monte Carlo Simulation:.*?Best PR-AUC Score\s*:\s*([\d.]+)', content, re.DOTALL)
        if mc_prauc_match:
            metrics['monte_carlo']['pr_auc'] = float(mc_prauc_match.group(1))
        
        # Extract Final Decision metrics
        final_single_f1_match = re.search(r'Single Model Option:.*?F1 Score\s*:\s*([-\d.]+)', content, re.DOTALL)
        if final_single_f1_match:
            metrics['final_decision']['single_f1'] = float(final_single_f1_match.group(1))
        
        final_ensemble_f1_match = re.search(r'Ensemble Option:.*?F1 Score\s*:\s*([\d.]+)', content, re.DOTALL)
        if final_ensemble_f1_match:
            metrics['final_decision']['ensemble_f1'] = float(final_ensemble_f1_match.group(1))
        
        # Determine which was selected
        if 'ENSEMBLE SELECTED' in content:
            metrics['final_decision']['selected'] = 'ensemble'
            metrics['final_decision']['selected_f1'] = metrics['final_decision'].get('ensemble_f1', 0)
        else:
            metrics['final_decision']['selected'] = 'single'
            metrics['final_decision']['selected_f1'] = metrics['final_decision'].get('single_f1', 0)
        
        return metrics
    
    def run_single_dataset(self, dataset_file: str, domain: str) -> Dict:
        """
        Run RAMSeS on a single dataset
        
        Parameters
        ----------
        dataset_file : str
            Dataset filename
        domain : str
            Domain name
            
        Returns
        -------
        Dict with results and metrics
        """
        logger.info("="*80)
        logger.info(f"STARTING PROCESSING: {dataset_file} from domain {domain}")
        logger.info("="*80)
        
        # Extract entity ID from filename
        # For SKAB: SKAB_3.csv -> 3
        # For SMD: machine-1-1.txt -> machine-1-1
        entity_match = re.search(r'_(\d+)\.csv$', dataset_file)
        if entity_match:
            entity = entity_match.group(1)
        else:
            # Remove extension (.csv, .txt, etc.)
            entity = re.sub(r'\.(csv|txt)$', '', dataset_file, flags=re.IGNORECASE)
        
        logger.info(f"  Dataset: {dataset_file}")
        logger.info(f"  Domain:  {domain}")
        logger.info(f"  Entity:  {entity}")
        
        # Check if results already exist (use lowercase domain for consistency)
        domain_lower = domain.lower()
        results_file = results_dir("comprehensive", domain_lower, entity) + f"comprehensive_results_{domain_lower}_{entity}_iter0.txt"
        logger.info(f"  Checking for existing results: {results_file}")
        
        if os.path.exists(results_file):
            logger.info(f"✓ Results already exist for {dataset_file}, skipping computation")
            logger.info(f"  Parsing existing results from: {results_file}")
            
            # Parse existing results
            metrics = self.parse_comprehensive_results(results_file)
            
            result = {
                'dataset_file': dataset_file,
                'domain': domain,
                'entity': entity,
                'total_runtime': metrics.get('timing', {}).get('end_to_end', 0),
                'avg_memory_mb': 0,  # Not available from file
                'peak_memory_mb': 0,  # Not available from file
                'metrics': metrics
            }
            
            logger.info(f"✓ COMPLETED (from cache): {dataset_file}")
            logger.info(f"  Runtime: {result['total_runtime']:.2f}s")
            logger.info("="*80)
            return result
        
        # Run app.py with dataset and entity arguments
        logger.info(f"→ No existing results found, running computation...")
        logger.info(f"→ Starting app.py for {domain}/{entity}...")
        
        # Build command
        cmd = [
            'python', 'app.py',
            '--dataset', domain.lower(),
            '--entity', entity,
            '--parallel', self.parallel
        ]
        
        logger.info(f"  Command: {' '.join(cmd)}")
        logger.info(f"  Timeout: {self.timeout}s")
        
        # Monitor memory usage
        memory_monitor = MemoryMonitor()
        memory_monitor.reset()
        
        start_time = time.time()
        logger.info(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        try:
            # Start subprocess
            logger.info("  → Launching subprocess...")
            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1  # Line buffered
            )
            
            logger.info(f"  → Process started (PID: {process.pid})")
            
            # Monitor memory while process runs and log output
            timeout_time = time.time() + self.timeout
            output_lines = []
            last_log_time = time.time()
            line_count = 0
            
            while process.poll() is None:
                memory_monitor.update()
                
                # Read and log output (non-blocking)
                line = process.stdout.readline()
                if line:
                    output_lines.append(line)
                    line_count += 1
                    
                    # Log important progress indicators (stages, generations, results)
                    if any(keyword in line for keyword in [
                        'STAGE', '🚀', '📂', '🔧', '💉', '📊', '🔍', '📝', '🎉', '✓',
                        'Generation', 'Loaded', 'trained models', 'Missing',
                        'F1 score', 'Evaluated fitness', 'Best ensemble',
                        'Sub-stage', 'Thompson Sampling', 'GAN', 'Monte Carlo',
                        'Final Decision', 'ENSEMBLE SELECTED', 'SINGLE MODEL SELECTED'
                    ]):
                        logger.info(f"    {line.strip()}")
                    
                    # Log error messages
                    if 'ERROR' in line or 'Error' in line or 'Traceback' in line:
                        logger.warning(f"    ⚠ {line.strip()}")
                
                # Periodic status update every 30 seconds
                current_time = time.time()
                if current_time - last_log_time >= 30:
                    elapsed = current_time - start_time
                    logger.info(f"  ⏱ Status: Running for {elapsed:.0f}s, Memory: {memory_monitor.get_peak():.1f} MB")
                    last_log_time = current_time
                
                # Check timeout
                if time.time() > timeout_time:
                    process.kill()
                    logger.error(f"✗ TIMEOUT: Process exceeded {self.timeout}s for {dataset_file}")
                    logger.error(f"  Last 10 lines of output:")
                    for line in output_lines[-10:]:
                        logger.error(f"    {line.rstrip()}")
                    return None
                
                # Only sleep if no output was received (to avoid busy-wait when idle)
                if not line:
                    time.sleep(0.01)  # Reduced from 0.1s to 0.01s (10ms)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"  → Process finished with return code: {process.returncode}")
            logger.info(f"  → Total lines captured: {line_count}")
            
            # Get any remaining output
            remaining_output = process.stdout.read()
            if remaining_output:
                output_lines.append(remaining_output)
            
            if process.returncode != 0:
                logger.error(f"✗ FAILED: app.py returned exit code {process.returncode} for {dataset_file}")
                logger.error(f"  Last 20 lines of output:")
                for line in output_lines[-20:]:
                    logger.error(f"    {line.rstrip()}")
                logger.info("="*80)
                return None
            
            logger.info(f"✓ EXECUTION COMPLETED: {dataset_file} in {total_time:.2f}s")
            
            # Parse the generated results
            logger.info(f"  → Parsing results from: {results_file}")
            if os.path.exists(results_file):
                metrics = self.parse_comprehensive_results(results_file)
                logger.info(f"  ✓ Results parsed successfully")
            else:
                logger.warning(f"✗ WARNING: Results file not found: {results_file}")
                logger.info("="*80)
                return None
            
            # Get memory stats
            avg_memory = memory_monitor.get_average()
            peak_memory = memory_monitor.get_peak()
            
            logger.info(f"  → Memory stats: Avg={avg_memory:.1f} MB, Peak={peak_memory:.1f} MB")
            
            result = {
                'dataset_file': dataset_file,
                'domain': domain,
                'entity': entity,
                'total_runtime': total_time,
                'avg_memory_mb': avg_memory,
                'peak_memory_mb': peak_memory,
                'metrics': metrics
            }
            
            logger.info(f"✓ COMPLETED SUCCESSFULLY: {dataset_file}")
            logger.info(f"  Total Runtime: {total_time:.2f}s")
            logger.info(f"  Peak Memory: {peak_memory:.1f} MB")
            logger.info("="*80)
            
            return result
            
        except Exception as e:
            logger.error(f"✗ EXCEPTION: Error processing {dataset_file}")
            logger.error(f"  Exception: {e}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            logger.info("="*80)
            return None
        
        return None
        
        # Find and parse comprehensive results
        results_file = results_dir("comprehensive", domain, entity) + f"comprehensive_results_{domain}_{entity}_iter0.txt"
        metrics = self.parse_comprehensive_results(results_file)
        
        # Add memory and total time info
        result = {
            'dataset_file': dataset_file,
            'domain': domain,
            'entity': entity,
            'total_runtime': metrics.get('timing', {}).get('end_to_end', 0),
            'avg_memory_mb': 0,  # Not available from existing results
            'peak_memory_mb': 0,  # Not available from existing results
            'metrics': metrics
        }
        
        return result
    
    def create_config(self, dataset: str, entity: str) -> Dict:
        """
        Create configuration for a specific dataset
        
        Parameters
        ----------
        dataset : str
            Dataset name
        entity : str
            Entity ID
            
        Returns
        -------
        Configuration dictionary
        """
        return {
            'dataset': dataset,
            'entity': entity,
            'iterations': 1,
            'anomaly_list': ['point', 'contextual'],
            'population_size': 20,
            'generations': 20,
            'meta_model_type': 'rf',
            'mutation_rate': 0.1
        }
    
    def run_domain(self, domain: str) -> List[Dict]:
        """
        Run all datasets for a specific domain
        
        Parameters
        ----------
        domain : str
            Domain name
            
        Returns
        -------
        List of results for all datasets in domain
        """
        logger.info("")
        logger.info("╔" + "="*78 + "╗")
        logger.info(f"║ STARTING DOMAIN: {domain:^60} ║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("")
        
        # Get datasets for this domain
        domain_datasets = self.datasets_df[self.datasets_df['domain_name'] == domain]['file_name'].tolist()
        
        logger.info(f"→ Found {len(domain_datasets)} datasets in domain '{domain}':")
        for i, ds in enumerate(domain_datasets, 1):
            logger.info(f"  {i:2d}. {ds}")
        logger.info("")
        
        results = []
        successful = 0
        failed = 0
        skipped = 0
        
        for idx, dataset_file in enumerate(domain_datasets, 1):
            logger.info(f"┌─ DATASET {idx}/{len(domain_datasets)} ─────────────────────────────────")
            logger.info(f"│  {dataset_file}")
            logger.info(f"└────────────────────────────────────────────────────────────")
            
            result = self.run_single_dataset(dataset_file, domain)
            if result:
                results.append(result)
                successful += 1
                self.save_intermediate_results(domain, results)
            else:
                failed += 1
            
            logger.info(f"  Progress: {idx}/{len(domain_datasets)} complete ({successful} success, {failed} failed)")
            logger.info("")
        
        logger.info("")
        logger.info("╔" + "="*78 + "╗")
        logger.info(f"║ DOMAIN COMPLETE: {domain:^58} ║")
        logger.info("╠" + "="*78 + "╣")
        logger.info(f"║  Total Datasets:  {len(domain_datasets):3d}                                                 ║")
        logger.info(f"║  Successful:      {successful:3d}                                                 ║")
        logger.info(f"║  Failed:          {failed:3d}                                                 ║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("")
        
        return results
    
    def save_intermediate_results(self, domain: str, results: List[Dict]):
        """Save intermediate results for a domain"""
        output_file = f"{self.output_base_dir}/{domain}/intermediate_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def compute_domain_statistics(self, domain_results: List[Dict]) -> Dict:
        """
        Compute aggregate statistics for a domain
        
        Parameters
        ----------
        domain_results : List[Dict]
            Results for all datasets in domain
            
        Returns
        -------
        Dict with aggregate statistics
        """
        if not domain_results:
            return {}
        
        stats = {
            'total_datasets': len(domain_results),
            'total_runtime': sum(r['total_runtime'] for r in domain_results),
            'avg_runtime': np.mean([r['total_runtime'] for r in domain_results]),
            'avg_memory_mb': np.mean([r['avg_memory_mb'] for r in domain_results]),
            'peak_memory_mb': np.mean([r['peak_memory_mb'] for r in domain_results]),
            'max_peak_memory_mb': max(r['peak_memory_mb'] for r in domain_results),
        }
        
        # Compute average module timings
        module_times = defaultdict(list)
        for result in domain_results:
            for module, time_val in result['metrics'].get('timing', {}).items():
                module_times[module].append(time_val)
        
        stats['avg_module_times'] = {
            module: np.mean(times) for module, times in module_times.items()
        }
        
        # Compute average F1 scores per module
        f1_scores = {
            'ga': [],
            'thompson': [],
            'gan': [],
            'borderline': [],
            'monte_carlo': [],
            'final_selected': []
        }
        
        for result in domain_results:
            metrics = result['metrics']
            if 'ga' in metrics and 'f1' in metrics['ga']:
                f1_scores['ga'].append(metrics['ga']['f1'])
            if 'thompson' in metrics and 'f1' in metrics['thompson']:
                f1_scores['thompson'].append(metrics['thompson']['f1'])
            if 'gan' in metrics and 'f1' in metrics['gan']:
                f1_scores['gan'].append(metrics['gan']['f1'])
            if 'borderline' in metrics and 'f1' in metrics['borderline']:
                f1_scores['borderline'].append(metrics['borderline']['f1'])
            if 'monte_carlo' in metrics and 'f1' in metrics['monte_carlo']:
                f1_scores['monte_carlo'].append(metrics['monte_carlo']['f1'])
            if 'final_decision' in metrics and 'selected_f1' in metrics['final_decision']:
                f1_scores['final_selected'].append(metrics['final_decision']['selected_f1'])
        
        stats['avg_f1_scores'] = {
            module: np.mean(scores) if scores else 0.0 
            for module, scores in f1_scores.items()
        }
        
        stats['std_f1_scores'] = {
            module: np.std(scores) if scores else 0.0 
            for module, scores in f1_scores.items()
        }
        
        return stats
    
    def generate_domain_report(self, domain: str, results: List[Dict], stats: Dict):
        """
        Generate comprehensive report for a domain
        
        Parameters
        ----------
        domain : str
            Domain name
        results : List[Dict]
            All results for domain
        stats : Dict
            Aggregate statistics
        """
        output_dir = f"{self.output_base_dir}/{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = f"{output_dir}/domain_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"RAMSeS Testbed Report - Domain: {domain}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Check if we have any stats
            if not stats:
                f.write("No datasets were successfully processed for this domain.\n")
                logger.warning(f"No statistics available for domain {domain}")
                return
            
            # Overall Statistics
            f.write("=" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Datasets Processed: {stats['total_datasets']}\n")
            f.write(f"Total Computational Time: {stats['total_runtime']:.2f}s ({stats['total_runtime']/60:.2f} min)\n")
            f.write(f"Average Runtime per Dataset: {stats['avg_runtime']:.2f}s\n")
            f.write(f"Average Memory Usage: {stats['avg_memory_mb']:.2f} MB\n")
            f.write(f"Average Peak Memory: {stats['peak_memory_mb']:.2f} MB\n")
            f.write(f"Maximum Peak Memory: {stats['max_peak_memory_mb']:.2f} MB\n\n")
            
            # Module Timing Breakdown
            f.write("=" * 80 + "\n")
            f.write("AVERAGE MODULE COMPUTATIONAL OVERHEAD\n")
            f.write("=" * 80 + "\n\n")
            
            for module, avg_time in sorted(stats['avg_module_times'].items()):
                if module != 'end_to_end':
                    f.write(f"  {module:<30s} : {avg_time:>10.4f}s\n")
            
            if 'end_to_end' in stats['avg_module_times']:
                f.write("-" * 50 + "\n")
                f.write(f"  {'End-to-End (Average)':<30s} : {stats['avg_module_times']['end_to_end']:>10.4f}s\n")
            f.write("\n")
            
            # F1 Scores
            f.write("=" * 80 + "\n")
            f.write("AVERAGE F1 SCORES BY MODULE\n")
            f.write("=" * 80 + "\n\n")
            
            for module, avg_f1 in stats['avg_f1_scores'].items():
                std_f1 = stats['std_f1_scores'].get(module, 0.0)
                f.write(f"  {module:<20s} : {avg_f1:.6f} ± {std_f1:.6f}\n")
            f.write("\n")
            
            # Per-Dataset Results
            f.write("=" * 80 + "\n")
            f.write("PER-DATASET RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for result in results:
                f.write(f"Dataset: {result['dataset_file']}\n")
                f.write(f"  Entity: {result['entity']}\n")
                f.write(f"  Runtime: {result['total_runtime']:.2f}s\n")
                f.write(f"  Avg Memory: {result['avg_memory_mb']:.2f} MB\n")
                f.write(f"  Peak Memory: {result['peak_memory_mb']:.2f} MB\n")
                
                # Module F1 scores
                metrics = result['metrics']
                if metrics:
                    f.write(f"  F1 Scores:\n")
                    if 'ga' in metrics and 'f1' in metrics['ga']:
                        f.write(f"    GA Ensemble: {metrics['ga']['f1']:.6f}\n")
                    if 'final_decision' in metrics and 'selected_f1' in metrics['final_decision']:
                        selected = metrics['final_decision']['selected']
                        f.write(f"    Final ({selected}): {metrics['final_decision']['selected_f1']:.6f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Domain report saved to: {report_file}")
    
    def generate_overall_summary(self, all_results: Dict[str, List[Dict]]):
        """
        Generate overall summary across all domains
        
        Parameters
        ----------
        all_results : Dict[str, List[Dict]]
            Results organized by domain
        """
        summary_file = f"{self.output_base_dir}/overall_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAMSeS Testbed - Overall Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            total_datasets = sum(len(results) for results in all_results.values())
            total_time = sum(
                sum(r['total_runtime'] for r in results)
                for results in all_results.values()
            )
            
            f.write(f"Total Domains: {len(all_results)}\n")
            f.write(f"Total Datasets: {total_datasets}\n")
            f.write(f"Total Computational Time: {total_time:.2f}s ({total_time/3600:.2f} hours)\n\n")
            
            # Per-domain summary
            f.write("=" * 80 + "\n")
            f.write("PER-DOMAIN SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for domain, results in all_results.items():
                if not results:
                    continue
                    
                stats = self.compute_domain_statistics(results)
                
                f.write(f"Domain: {domain}\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Datasets: {len(results)}\n")
                f.write(f"  Total Time: {stats['total_runtime']:.2f}s\n")
                f.write(f"  Avg Time: {stats['avg_runtime']:.2f}s\n")
                f.write(f"  Avg Memory: {stats['avg_memory_mb']:.2f} MB\n")
                f.write(f"  Avg Peak Memory: {stats['peak_memory_mb']:.2f} MB\n")
                f.write(f"  Avg F1 (Final): {stats['avg_f1_scores'].get('final_selected', 0):.6f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Overall summary saved to: {summary_file}")
        
        # Also save as JSON
        json_summary = {
            'total_domains': len(all_results),
            'total_datasets': total_datasets,
            'total_time': total_time,
            'domains': {}
        }
        
        for domain, results in all_results.items():
            if results:
                stats = self.compute_domain_statistics(results)
                json_summary['domains'][domain] = stats
        
        json_file = f"{self.output_base_dir}/overall_summary.json"
        with open(json_file, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        logger.info(f"Overall summary (JSON) saved to: {json_file}")
    
    def run_all(self):
        """Run testbed on all domains"""
        logger.info("=" * 80)
        logger.info("Starting RAMSeS Comprehensive Testbed")
        logger.info("=" * 80)
        
        all_results = {}
        
        for domain in self.domains:
            domain_results = self.run_domain(domain)
            all_results[domain] = domain_results
            
            # Generate domain-specific report
            stats = self.compute_domain_statistics(domain_results)
            self.generate_domain_report(domain, domain_results, stats)
        
        # Generate overall summary
        self.generate_overall_summary(all_results)
        
        logger.info("=" * 80)
        logger.info("Testbed Complete!")
        logger.info(f"Results saved in: {self.output_base_dir}/")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAMSeS Comprehensive Testbed')
    parser.add_argument(
        '--dataset-list',
        type=str,
        default='testbed/file_list/test_m_skab.csv',
        help='Path to dataset list CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='testbed_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout in seconds for each dataset (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--domain',
        type=str,
        default=None,
        help='Run only specific domain (optional)'
    )
    parser.add_argument(
        '--parallel',
        type=str,
        default='false',
        help='Run model selection in parallel mode (true/false, t/f, T/F)'
    )
    
    args = parser.parse_args()
    
    # Create testbed runner
    runner = TestbedRunner(args.dataset_list, args.output_dir, args.timeout, args.parallel)
    
    if args.domain:
        # Run single domain
        logger.info(f"Running single domain: {args.domain}")
        results = runner.run_domain(args.domain)
        stats = runner.compute_domain_statistics(results)
        runner.generate_domain_report(args.domain, results, stats)
    else:
        # Run all domains
        runner.run_all()


if __name__ == '__main__':
    main()
