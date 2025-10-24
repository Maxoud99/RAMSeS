# Metrics/Ens_GA.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Union
from random import sample
from Utils.model_selection_utils import evaluate_model
from copy import deepcopy
from Datasets.dataset import Entity, Dataset
from loguru import logger
# Metrics/Ens_GA.py


import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score
from random import sample, uniform, randint
from Datasets.dataset import Entity, Dataset
from loguru import logger

def generate_initial_population(models: List[str], population_size: int = 10) -> List:
    """Generate initial population of ensembles with random subsets of models and random weights."""
    population = []
    for _ in range(population_size):
        subset_size = randint(1, len(models))  # Random subset size
        subset = sample(models, subset_size)  # Random subset of models
        individual = {model: uniform(0, 1) for model in subset}
        # Normalize weights
        total_weight = sum(individual.values())
        individual = {k: v / total_weight for k, v in individual.items()}
        population.append(individual)
    return population

def evaluate_ensemble(data: Union[Dataset, Entity], ensemble, models_dict: Dict[str, 'PyMADModel'],
                      eval_batch_size: int = 128) -> dict:
    """Evaluate an ensemble of models on the data using PR_AUC or F1 score."""
    predictions = []
    for model_name, weight in ensemble.items():
        model = models_dict[model_name]
        model_predictions = evaluate_model(data=data, model=model, model_name=model_name, padding_type='right',
                                           eval_batch_size=eval_batch_size)
        predictions.append(model_predictions['entity_scores'] * weight)

    ensemble_scores = sum(predictions) / len(predictions)

    # Flatten ensemble_scores and y_true
    y_true = data.entities[0].labels.flatten()
    ensemble_scores = ensemble_scores.flatten()

    precision, recall, _ = precision_recall_curve(y_true, ensemble_scores)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, (ensemble_scores > 0.5).astype(int))

    return {'pr_auc': pr_auc, 'f1': f1}

def genetic_algorithm(models_dict: Dict[str, 'PyMADModel'], train_data, test_data, num_generations: int = 10,
                      population_size: int = 10, metric: str = 'pr_auc') -> List:
    """Genetic algorithm to optimize model ensembles."""
    model_names = list(models_dict.keys())
    population = generate_initial_population(model_names, population_size)

    for generation in range(num_generations):
        logger.info(f"Generation {generation + 1}/{num_generations}")
        scores = []
        for ensemble in population:
            test_sample_indices = np.random.choice(len(test_data.entities[0].Y[0]),
                                                   len(test_data.entities[0].Y[0]) // 10, replace=False)
            test_sample = Entity(Y=test_data.entities[0].Y[:, test_sample_indices],
                                 labels=test_data.entities[0].labels[:, test_sample_indices])
            test_sample_data = Dataset(entities=[test_sample], name="test_sample")
            result = evaluate_ensemble(data=test_sample_data, ensemble=ensemble, models_dict=models_dict)
            scores.append((ensemble, result[metric]))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_individuals = [x[0] for x in scores[:population_size // 2]]

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = sample(top_individuals, 2)
            crossover_point = randint(1, len(model_names) - 1)

            child1_keys = list(set(list(parent1.keys())[:crossover_point] + list(parent2.keys())[crossover_point:]))
            child2_keys = list(set(list(parent2.keys())[:crossover_point] + list(parent1.keys())[crossover_point:]))

            child1 = {key: uniform(0, 1) for key in child1_keys}
            child2 = {key: uniform(0, 1) for key in child2_keys}

            # Mutation
            for child in [child1, child2]:
                if uniform(0, 1) < 0.1:  # 10% chance of mutation
                    model_to_mutate = sample(model_names, 1)[0]
                    if model_to_mutate in child:
                        child[model_to_mutate] = uniform(0, 1)
                    else:
                        child[model_to_mutate] = uniform(0, 1)

                # Normalize weights after mutation
                total_weight = sum(child.values())
                child = {k: v / total_weight for k, v in child.items()}

            new_population.extend([child1, child2])

        population = top_individuals + new_population[:population_size - len(top_individuals)]

    final_scores = [(ensemble, evaluate_ensemble(test_data, ensemble, models_dict)[metric]) for ensemble in population]
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores
