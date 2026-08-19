"""
Results Formatter Module
Provides organized output formatting and overhead analysis for RAMSeS framework
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

class ResultsFormatter:
    """Handles formatting and saving of model selection results with timing analysis"""
    
    def __init__(self, dataset: str, entity: str, output_dir: str = "myresults"):
        self.dataset = dataset
        self.entity = entity
        self.output_dir = output_dir
        self.timings = {}
        self.results = {}
        self.start_time = time.time()
        
    def start_timer(self, component: str):
        """Start timing a component"""
        self.timings[f"{component}_start"] = time.time()
        
    def end_timer(self, component: str):
        """End timing a component and store duration"""
        if f"{component}_start" in self.timings:
            duration = time.time() - self.timings[f"{component}_start"]
            self.timings[f"{component}_duration"] = duration
            return duration
        return 0
    
    def add_result(self, component: str, data: Any):
        """Store results for a component"""
        self.results[component] = data
    
    def get_overhead_summary(self) -> Dict[str, float]:
        """Calculate overhead summary"""
        overhead = {}
        for key, value in self.timings.items():
            if key.endswith('_duration'):
                component = key.replace('_duration', '')
                overhead[component] = value
        
        total_time = time.time() - self.start_time
        overhead['total'] = total_time
        overhead['other'] = total_time - sum([v for k, v in overhead.items() if k != 'total'])
        
        return overhead
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.2f}s"
    
    def save_comprehensive_results(self):
        """Save comprehensive formatted results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(self.output_dir, "comprehensive", self.dataset, str(self.entity))
        os.makedirs(base_dir, exist_ok=True)
        
        # 1. Save human-readable text report
        text_file = os.path.join(base_dir, f"results_{timestamp}.txt")
        self._save_text_report(text_file)

        # 2. Save JSON with all data.
        json_file = os.path.join(base_dir, f"results_{timestamp}.json")
        self._save_json_report(json_file)
        
        # 3. Save overhead analysis
        overhead_file = os.path.join(base_dir, f"overhead_{timestamp}.txt")
        self._save_overhead_analysis(overhead_file)
        
        return text_file, json_file, overhead_file
    
    def _save_text_report(self, filepath: str):
        """Save formatted text report"""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAMSeS - Robust & Adaptive Model Selection Framework\n")
            f.write("COMPREHENSIVE RESULTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Entity: {self.entity}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runtime: {self.format_duration(time.time() - self.start_time)}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Section 1: Genetic Algorithm Results
            if 'ga' in self.results:
                f.write("1. GENETIC ALGORITHM (GA) - Ensemble Selection\n")
                f.write("-" * 80 + "\n")
                ga_data = self.results['ga']
                f.write(f"Best Ensemble Size: {len(ga_data['best_ensemble'])}\n")
                f.write(f"Best Ensemble Models: {', '.join(ga_data['best_ensemble'])}\n")
                f.write(f"F1 Score: {ga_data['f1']:.6f}\n")
                f.write(f"PR-AUC: {ga_data['pr_auc']:.6f}\n")
                f.write(f"Fitness: {ga_data['fitness']:.6f}\n")
                if 'ga_duration' in self.timings:
                    f.write(f"Runtime: {self.format_duration(self.timings['ga_duration'])}\n")
                f.write("\n")
            
            # Section 2: Thompson Sampling Results
            if 'thompson' in self.results:
                f.write("2. THOMPSON SAMPLING - Online Model Selection\n")
                f.write("-" * 80 + "\n")
                thompson_data = self.results['thompson']
                f.write("Top-10 Models (by selection frequency):\n")
                for i, model in enumerate(thompson_data['top_models'][:10], 1):
                    f.write(f"  {i:2d}. {model}\n")
                
                # Add F1 and PR-AUC metrics if available
                if 'f1' in thompson_data and 'pr_auc' in thompson_data:
                    f.write(f"\nTop Model Performance:\n")
                    f.write(f"  F1 Score: {thompson_data['f1']:.6f}\n")
                    f.write(f"  PR-AUC: {thompson_data['pr_auc']:.6f}\n")
                
                if 'thompson_duration' in self.timings:
                    f.write(f"Runtime: {self.format_duration(self.timings['thompson_duration'])}\n")
                f.write("\n")
            
            # Section 3: Robustness Tests
            f.write("3. ROBUSTNESS TESTING RESULTS\n")
            f.write("-" * 80 + "\n\n")
            
            # 3a. GAN Test
            if 'gan' in self.results:
                f.write("3a. GAN Robustness Test\n")
                f.write("    " + "-" * 76 + "\n")
                gan_data = self.results['gan']
                f.write("    Top-5 by F1 Score:\n")
                for i, model in enumerate(gan_data['f1_ranking'][:5], 1):
                    f.write(f"      {i}. {model}\n")
                f.write("\n    Top-5 by PR-AUC:\n")
                for i, model in enumerate(gan_data['pr_ranking'][:5], 1):
                    f.write(f"      {i}. {model}\n")
                
                # Add best F1 and PR-AUC metrics
                if 'f1' in gan_data and 'pr_auc' in gan_data:
                    f.write(f"\n    Best Model Performance:\n")
                    f.write(f"      F1 Score: {gan_data['f1']:.6f}\n")
                    f.write(f"      PR-AUC: {gan_data['pr_auc']:.6f}\n")
                
                if 'gan_duration' in self.timings:
                    f.write(f"    Runtime: {self.format_duration(self.timings['gan_duration'])}\n")
                f.write("\n")
            
            # 3b. Borderline Sensitivity
            if 'borderline' in self.results:
                f.write("3b. Borderline Sensitivity Test\n")
                f.write("    " + "-" * 76 + "\n")
                borderline_data = self.results['borderline']
                f.write("    Top-5 by F1 Score:\n")
                for i, model in enumerate(borderline_data['f1_ranking'][:5], 1):
                    f.write(f"      {i}. {model}\n")
                f.write("\n    Top-5 by PR-AUC:\n")
                for i, model in enumerate(borderline_data['pr_ranking'][:5], 1):
                    f.write(f"      {i}. {model}\n")
                
                # Add best F1 and PR-AUC metrics
                if 'f1' in borderline_data and 'pr_auc' in borderline_data:
                    f.write(f"\n    Best Model Performance:\n")
                    f.write(f"      F1 Score: {borderline_data['f1']:.6f}\n")
                    f.write(f"      PR-AUC: {borderline_data['pr_auc']:.6f}\n")
                
                if 'borderline_duration' in self.timings:
                    f.write(f"    Runtime: {self.format_duration(self.timings['borderline_duration'])}\n")
                f.write("\n")
            
            # 3c. Monte Carlo
            if 'montecarlo' in self.results:
                f.write("3c. Monte Carlo Simulation\n")
                f.write("    " + "-" * 76 + "\n")
                mc_data = self.results['montecarlo']
                f.write("    Top-5 by F1 Score:\n")
                for i, model in enumerate(mc_data['f1_ranking'][:5], 1):
                    f.write(f"      {i}. {model}\n")
                f.write("\n    Top-5 by PR-AUC:\n")
                for i, model in enumerate(mc_data['pr_ranking'][:5], 1):
                    f.write(f"      {i}. {model}\n")
                
                # Add best F1 and PR-AUC metrics (mean values for Monte Carlo)
                if 'f1' in mc_data and 'pr_auc' in mc_data:
                    f.write(f"\n    Best Model Performance (Mean):\n")
                    f.write(f"      F1 Score: {mc_data['f1']:.6f}\n")
                    f.write(f"      PR-AUC: {mc_data['pr_auc']:.6f}\n")
                
                if 'montecarlo_duration' in self.timings:
                    f.write(f"    Runtime: {self.format_duration(self.timings['montecarlo_duration'])}\n")
                f.write("\n")
            
            # Section 4: Aggregated Rankings
            f.write("4. AGGREGATED RANKINGS\n")
            f.write("-" * 80 + "\n\n")
            
            if 'robust_aggregate' in self.results:
                f.write("4a. Robust Tests Aggregate (GAN + Borderline + Monte Carlo)\n")
                f.write("    " + "-" * 76 + "\n")
                f.write("    Top-10 Models:\n")
                for i, model in enumerate(self.results['robust_aggregate'][:10], 1):
                    f.write(f"      {i:2d}. {model}\n")
                f.write("\n")
            
            if 'final_aggregate' in self.results:
                f.write("4b. Final Aggregate (Robust + Thompson Sampling)\n")
                f.write("    " + "-" * 76 + "\n")
                f.write("    Complete Ranking (All 27 Models):\n")
                for i, model in enumerate(self.results['final_aggregate'], 1):
                    f.write(f"      {i:2d}. {model}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
    
    def _save_json_report(self, filepath: str):
        """Save complete results as JSON"""
        data = {
            'metadata': {
                'dataset': self.dataset,
                'entity': self.entity,
                'timestamp': datetime.now().isoformat(),
                'total_runtime_seconds': time.time() - self.start_time,
            },
            'results': self.results,
            'timings': {k: v for k, v in self.timings.items() if k.endswith('_duration')},
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_overhead_analysis(self, filepath: str):
        """Save detailed overhead analysis"""
        overhead = self.get_overhead_summary()
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OVERHEAD ANALYSIS - Per Module Runtime\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset} | Entity: {self.entity}\n")
            f.write(f"Total End-to-End Runtime: {self.format_duration(overhead['total'])}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Per-Module Breakdown:\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort by duration descending
            sorted_components = sorted(
                [(k, v) for k, v in overhead.items() if k not in ['total', 'other']],
                key=lambda x: x[1],
                reverse=True
            )
            
            for component, duration in sorted_components:
                percentage = (duration / overhead['total']) * 100
                f.write(f"{component:25s}: {self.format_duration(duration):>12s}  ({percentage:5.2f}%)\n")
            
            if overhead['other'] > 0:
                percentage = (overhead['other'] / overhead['total']) * 100
                f.write(f"{'Other/Overhead':25s}: {self.format_duration(overhead['other']):>12s}  ({percentage:5.2f}%)\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Visual bar chart
            f.write("Visual Breakdown:\n")
            f.write("-" * 80 + "\n\n")
            
            max_bar_width = 60
            for component, duration in sorted_components:
                percentage = (duration / overhead['total']) * 100
                bar_length = int((duration / overhead['total']) * max_bar_width)
                bar = '█' * bar_length
                f.write(f"{component:15s} |{bar:<{max_bar_width}}| {percentage:5.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    def print_summary(self):
        """Print a quick summary to console"""
        print("\n" + "=" * 80)
        print("RAMSES EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Dataset: {self.dataset} | Entity: {self.entity}")
        print(f"Total Runtime: {self.format_duration(time.time() - self.start_time)}")
        
        if 'final_aggregate' in self.results:
            print(f"\nTop-5 Models (Final Ranking):")
            for i, model in enumerate(self.results['final_aggregate'][:5], 1):
                print(f"  {i}. {model}")
        
        overhead = self.get_overhead_summary()
        print(f"\nModule Runtimes:")
        for component in ['ga', 'thompson', 'gan', 'borderline', 'montecarlo']:
            if f'{component}_duration' in self.timings:
                print(f"  {component:12s}: {self.format_duration(self.timings[f'{component}_duration'])}")
        
        print("=" * 80 + "\n")
