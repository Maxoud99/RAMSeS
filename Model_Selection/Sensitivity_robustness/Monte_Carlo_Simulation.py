import numpy as np
import copy
import os
from Metrics.metrics import range_based_precision_recall_f1_auc, prauc, f1_score, f1_soft_score
from Utils.model_selection_utils import evaluate_model
from loguru import logger
import matplotlib.pyplot as plt


def add_noise_to_data(data, noise_level=0.1):
    """Add Gaussian noise to the data."""
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise


def monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations=100, noise_level=0.1):
    """
    Perform Monte Carlo simulation to assess model robustness.

    Args:
        test_data: Original test data.
        trained_models: Dictionary of trained models.
        model_names: List of model names.
        dataset: Dataset name.
        entity: Entity name.
        n_simulations: Number of Monte Carlo simulations.
        noise_level: Level of noise to add to the data.

    Returns:
        A dictionary containing aggregated performance metrics for each model.
    """
    results = {model_name: {'f1_scores': [], 'pr_auc_scores': []} for model_name in model_names}

    for sim in range(n_simulations):
        logger.info(f"Simulation {sim + 1}/{n_simulations}")
        noisy_data = copy.deepcopy(test_data)
        noisy_data.entities[0].Y = add_noise_to_data(noisy_data.entities[0].Y, noise_level)

        for model_name in model_names:
            model = trained_models.get(model_name)
            if model:
                evaluation = evaluate_model(noisy_data, model, model_name)
                y_true = evaluation['anomaly_labels'].flatten()
                y_scores = evaluation['entity_scores'].flatten()
                _, _, f1, pr_auc, _ = range_based_precision_recall_f1_auc(y_true, y_scores)
                # f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true)
                # pr_auc = prauc(y_true, y_scores)
                results[model_name]['f1_scores'].append(f1)
                results[model_name]['pr_auc_scores'].append(pr_auc)

    return results


def summarize_results(results):
    """Summarize the results of Monte Carlo simulation and rank models."""
    summary = {}
    for model_name, metrics in results.items():
        f1_mean = np.mean(metrics['f1_scores'])
        f1_std = np.std(metrics['f1_scores'])
        pr_auc_mean = np.mean(metrics['pr_auc_scores'])
        pr_auc_std = np.std(metrics['pr_auc_scores'])
        summary[model_name] = {
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'pr_auc_mean': pr_auc_mean,
            'pr_auc_std': pr_auc_std
        }

    # Rank models by F1 score
    ranked_by_f1 = sorted(summary.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
    ranked_by_pr_auc = sorted(summary.items(), key=lambda x: x[1]['pr_auc_mean'], reverse=True)

    summary['ranked_by_f1'] = [item[0] for item in ranked_by_f1]
    summary['ranked_by_pr_auc'] = [item[0] for item in ranked_by_pr_auc]

    return summary


def plot_monte_carlo_results(results, summary, model_names, dataset, entity):
    """Plot the results of Monte Carlo simulation."""
    for model_name in model_names:
        f1_scores = results[model_name]['f1_scores']
        pr_auc_scores = results[model_name]['pr_auc_scores']

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(f1_scores, bins=20, alpha=0.7, color='blue')
        plt.title(f'{model_name} F1 Scores')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.axvline(summary[model_name]['f1_mean'], color='red', linestyle='dashed', linewidth=2)
        plt.text(summary[model_name]['f1_mean'] + summary[model_name]['f1_std'], max(plt.ylim()) * 0.9,
                 f'Mean: {summary[model_name]["f1_mean"]:.2f}\nStd: {summary[model_name]["f1_std"]:.2f}',
                 color='red')

        plt.subplot(1, 2, 2)
        plt.hist(pr_auc_scores, bins=20, alpha=0.7, color='green')
        plt.title(f'{model_name} PR AUC Scores')
        plt.xlabel('PR AUC Score')
        plt.ylabel('Frequency')
        plt.axvline(summary[model_name]['pr_auc_mean'], color='red', linestyle='dashed', linewidth=2)
        plt.text(summary[model_name]['pr_auc_mean'] + summary[model_name]['pr_auc_std'], max(plt.ylim()) * 0.9,
                 f'Mean: {summary[model_name]["pr_auc_mean"]:.2f}\nStd: {summary[model_name]["pr_auc_std"]:.2f}',
                 color='red')

        plt.tight_layout()

        # Save the plot
        directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
        os.makedirs(directory, exist_ok=True)
        filename = f'{dataset}_{entity}_{model_name}_MonteCarloResults.png'
        plt.savefig(os.path.join(directory, filename), dpi=300)

        plt.show()


def save_summary(summary, dataset, entity):
    """Save the summary of Monte Carlo simulation to a file."""
    directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    summary_file = os.path.join(directory, f'{dataset}_{entity}_MonteCarloSummary.txt')

    with open(summary_file, 'w') as f:
        f.write("Summary of Monte Carlo Simulation:\n")
        for model_name, metrics in summary.items():
            if model_name not in ['ranked_by_f1', 'ranked_by_pr_auc']:
                f.write(f"Model: {model_name}\n")
                f.write(f"  F1 Mean: {metrics['f1_mean']:.4f}, F1 Std: {metrics['f1_std']:.4f}\n")
                f.write(f"  PR AUC Mean: {metrics['pr_auc_mean']:.4f}, PR AUC Std: {metrics['pr_auc_std']:.4f}\n")

        f.write("\nModels ranked by F1 score:\n")
        for rank, model_name in enumerate(summary['ranked_by_f1'], 1):
            f.write(f"{rank}. {model_name}\n")

        f.write("\nModels ranked by PR AUC score:\n")
        for rank, model_name in enumerate(summary['ranked_by_pr_auc'], 1):
            f.write(f"{rank}. {model_name}\n")


def run_monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations=100,
                               noise_level=0.1):
    """Run the entire Monte Carlo simulation process."""
    # Run Monte Carlo simulation
    results = monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations,
                                     noise_level)

    # Summarize results
    summary = summarize_results(results)

    # Print summary and rankings
    print("Summary of Monte Carlo Simulation:")
    for model_name, metrics in summary.items():
        if model_name not in ['ranked_by_f1', 'ranked_by_pr_auc']:
            print(f"Model: {model_name}")
            print(f"  F1 Mean: {metrics['f1_mean']:.4f}, F1 Std: {metrics['f1_std']:.4f}")
            print(f"  PR AUC Mean: {metrics['pr_auc_mean']:.4f}, PR AUC Std: {metrics['pr_auc_std']:.4f}")

    print("\nModels ranked by F1 score:")
    ranked_models_F1 = []
    for rank, model_name in enumerate(summary['ranked_by_f1'], 1):
        print(f"{rank}. {model_name}")
        ranked_models_F1.append(model_name)
    ranked_models_PR = []
    print("\nModels ranked by PR AUC score:")
    for rank, model_name in enumerate(summary['ranked_by_pr_auc'], 1):
        print(f"{rank}. {model_name}")
        ranked_models_PR.append(model_name)

    # Save summary
    save_summary(summary, dataset, entity)

    # Plot results
    plot_monte_carlo_results(results, summary, model_names, dataset, entity)
    return ranked_models_F1, ranked_models_PR



