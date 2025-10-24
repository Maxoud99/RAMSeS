# time_series_framework/Ensemble_Genetics.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from typing import List, Union
from copy import deepcopy
import torch as t
from typing import Tuple, Union, List
import numpy as np
import torch as t
from tqdm import tqdm, trange
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import ParameterGrid

from Loaders.loader import Loader
from Datasets.dataset import Dataset, Entity
from Algorithms.base_model import PyMADModel

from Utils.utils import de_unfold
from Model_Training.hyperparameter_grids import *

# Placeholder for your data loader, model, and other dependencies
import Datasets

def predict(batch: dict, model_name: str,
            model: PyMADModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a model and an input batch (Y), use the model to predict Y_hat.

    Parameters
    ----------
    batch: dict
        Input batch
    model_name: str
        Name of the model to use for prediction
    """
    _VALID_MODEL_NAMES = ['RNN', 'DGHL', 'LSTMVAE', 'MD', 'RM',
                          'NN']  # TODO: Should be stored somewhere centrally

    model_type = model_name.split('_')[0]

    if model_type == 'RNN':
        return _predict_rnn(batch, model)
    elif model_type == 'DGHL':
        return _predict_dghl(batch, model)
    elif model_type == 'NN':
        return _predict_nn(batch, model)
    elif model_type == 'MD':
        return _predict_md(batch, model)
    elif model_type == 'RM':
        return _predict_rm(batch, model)
    elif model_type == 'LSTMVAE':
        return _predict_lstmvae(batch, model)
    elif model_type == 'LOF':
        return _predict_lof(batch, model)
    elif model_type == 'KDE':
        return _predict_kde(batch, model)
    elif model_type == 'ABOD':
        return _predict_abod(batch, model)
    elif model_type == 'CBLOF':
        return _predict_cblof(batch, model)
    elif model_type == 'COF':
        return _predict_cof(batch, model)
    elif model_type == 'SOS':
        return _predict_sos(batch, model)
    elif model_type == 'ALAD':
        return _predict_alad(batch, model)
    else:
        return  _predict_pyod(batch, model)





import numpy as np
import torch as t
from typing import Union, Tuple

def pad_to_max_shape(predictions):
    """Pad all arrays to the maximum shape found in the predictions."""
    max_shape = tuple(max(sizes) for sizes in zip(*[pred.shape for pred in predictions]))
    padded_predictions = []
    for pred in predictions:
        pad_width = [(0, max_dim - curr_dim) for curr_dim, max_dim in zip(pred.shape, max_shape)]
        padded_pred = np.pad(pred, pad_width, mode='constant')
        padded_predictions.append(padded_pred)
    return padded_predictions


def evaluate_model(data: Union[Dataset, Entity],
                   model: t.nn.Module,
                   model_name: str,
                   padding_type: str = 'right',
                   eval_batch_size: int = 128) -> dict:
    """Compute observations necessary to evaluate a model on a given dataset."""
    anomaly_labels = data.entities[0].labels

    dataloader = Loader(dataset=data,
                        batch_size=eval_batch_size,
                        window_size=model.window_size,
                        window_step=model.window_step,
                        shuffle=False,
                        padding_type=padding_type,
                        sample_with_replace=False,
                        verbose=False,
                        mask_position='None',
                        n_masked_timesteps=0)

    if model.window_size == -1:
        window_size = data.entities[0].Y.shape[1]
    else:
        window_size = model.window_size

    entity_scores = t.zeros((len(dataloader) * eval_batch_size, data.n_features, window_size))

    n_features = data.n_features
    if 'DGHL' in model_name and (
            data.entities[0].X is not None):  # DGHL also considers covariates
        n_features = n_features + data.entities[0].X.shape[0]

    # Adjust the shapes to ensure they match the predictions
    # Initialize with None and dynamically set based on first batch
    Y = None
    Y_hat = None
    Y_sigma = None
    mask = None

    step = 0
    for batch in dataloader:
        batch_size, n_features, window_size = batch['Y'].shape

        try:
            # Entity anomaly scores to compute PR-AUC and Centrality
            batch_anomaly_score = model.window_anomaly_score(input=batch,
                                                             return_detail=True)
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            if batch_anomaly_score.shape[0] != batch_size:
                print(
                    f"Shape mismatch: batch_anomaly_score shape: {batch_anomaly_score.shape}, expected shape: {entity_scores[start_idx:end_idx, :, :].shape}")

            entity_scores[start_idx:end_idx, :, :] = batch_anomaly_score.detach()

            # Forecasting Error
            Y_b, Y_hat_b, Y_sigma_b, mask_b = predict(batch, model_name, model)

            print(f"Y_b shape: {Y_b.shape}, Y_hat_b shape: {Y_hat_b.shape}, Y_sigma_b shape: {Y_sigma_b.shape}, mask_b shape: {mask_b.shape}")

            if Y is None:  # Initialize the arrays based on the first batch
                total_batches = len(dataloader)
                Y = np.zeros((total_batches * batch_size, n_features, window_size))
                Y_hat = np.zeros((total_batches * batch_size, n_features, window_size))
                Y_sigma = np.zeros((total_batches * batch_size, n_features, window_size))
                mask = np.zeros((total_batches * batch_size, n_features, window_size))

            def pad_or_trim(array, target_shape):
                """Pad or trim array to match the target shape."""
                current_shape = array.shape
                if current_shape == target_shape:
                    return array
                pad_width = [(0, max(0, t - s)) for s, t in zip(current_shape, target_shape)]
                trimmed_array = array[tuple(slice(0, min(s, t)) for s, t in zip(current_shape, target_shape))]
                return np.pad(trimmed_array, pad_width, mode='constant')

            Y_b = pad_or_trim(Y_b, (batch_size, n_features, window_size))
            Y_hat_b = pad_or_trim(Y_hat_b, (batch_size, n_features, window_size))
            Y_sigma_b = pad_or_trim(Y_sigma_b, (batch_size, n_features, window_size))
            mask_b = pad_or_trim(mask_b, (batch_size, n_features, window_size))

            Y[start_idx:end_idx, :, :] = Y_b
            Y_hat[start_idx:end_idx, :, :] = Y_hat_b
            Y_sigma[start_idx:end_idx, :, :] = Y_sigma_b
            mask[start_idx:end_idx, :, :] = mask_b

        except Exception as e:
            print(f"Error processing batch: {e}")
            print(f"batch_size: {batch_size}, n_features: {n_features}, window_size: {window_size}")
            if 'Y_b' in locals():
                print(
                    f"Y_b shape: {Y_b.shape}, Y_hat_b shape: {Y_hat_b.shape}, Y_sigma_b shape: {Y_sigma_b.shape}, mask_b shape: {mask_b.shape}")
            else:
                print("Y_b is not defined due to an earlier error")
            raise

        step += 1

    # Final Anomaly Scores and forecasts
    entity_scores = model.final_anomaly_score(
        input=entity_scores, return_detail=False
    )  # return_detail = False averages the anomaly scores across features.
    entity_scores = entity_scores.detach().cpu().numpy()

    Y_hat = de_unfold(windows=Y_hat, window_step=model.window_step)
    Y = de_unfold(windows=Y, window_step=model.window_step)
    Y_sigma = de_unfold(windows=Y_sigma, window_step=model.window_step)
    mask = de_unfold(windows=mask, window_step=model.window_step)

    # Remove extra padding from Anomaly Scores and forecasts
    entity_scores = _adjust_scores_with_padding(
        scores=entity_scores,
        padding_size=dataloader.padding_size,
        padding_type=padding_type)

    Y_hat = _adjust_scores_with_padding(scores=Y_hat,
                                        padding_size=dataloader.padding_size,
                                        padding_type=padding_type)
    Y = _adjust_scores_with_padding(scores=Y,
                                    padding_size=dataloader.padding_size,
                                    padding_type=padding_type)
    Y_sigma = _adjust_scores_with_padding(scores=Y_sigma,
                                          padding_size=dataloader.padding_size,
                                          padding_type=padding_type)
    mask = _adjust_scores_with_padding(scores=mask,
                                       padding_size=dataloader.padding_size,
                                       padding_type=padding_type)

    return {
        'entity_scores': entity_scores,
        'Y': Y,
        'Y_hat': Y_hat,
        'Y_sigma': Y_sigma,
        'mask': mask,
        'anomaly_labels': anomaly_labels  # Ensure this is a flat array
    }


######################################################
# Ensemble Methods
######################################################
def bagging_ensemble(models: List[PyMADModel], data: Union[Dataset, Entity], eval_batch_size: int = 128) -> dict:
    """Evaluate a bagging ensemble of models."""
    predictions = []
    for model in models:
        try:
            preds = evaluate_model(data, model, model_name=str(model), eval_batch_size=eval_batch_size)
            predictions.append(preds['entity_scores'])
        except ValueError as ve:
            print(f"Skipping model due to evaluation error: {ve}")

    if not predictions:
        raise ValueError("No valid predictions were generated. Please check the models and data consistency.")

    # Pad predictions to have the same shape
    predictions = pad_to_max_shape(predictions)

    # Aggregate predictions (mean for bagging)
    entity_scores = np.mean(predictions, axis=0)

    return {
        'entity_scores': entity_scores
    }

def boosting_ensemble(models: List[PyMADModel], data: Union[Dataset, Entity], eval_batch_size: int = 128) -> dict:
    """Evaluate a boosting ensemble of models."""
    predictions = []
    for model in models:
        try:
            preds = evaluate_model(data, model, model_name=str(model), eval_batch_size=eval_batch_size)
            predictions.append(preds['entity_scores'])
        except ValueError as ve:
            print(f"Skipping model due to evaluation error: {ve}")

    if not predictions:
        raise ValueError("No valid predictions were generated. Please check the models and data consistency.")

    # Pad predictions to have the same shape
    predictions = pad_to_max_shape(predictions)

    # Aggregate predictions (mean for boosting)
    entity_scores = np.mean(predictions, axis=0)

    return {
        'entity_scores': entity_scores
    }

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Union

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import List, Union

def stacking_ensemble(models: List[PyMADModel], data: Union[Dataset, Entity], meta_model=None, eval_batch_size: int = 128) -> dict:
    """Evaluate a stacking ensemble of models."""
    if meta_model is None:
        meta_model = LogisticRegression()

    predictions = []
    targets = []

    for model in models:
        try:
            preds = evaluate_model(data, model, model_name=str(model), eval_batch_size=eval_batch_size)
            predictions.append(preds['entity_scores'])
            targets.append(preds['anomaly_labels'])
        except ValueError as ve:
            print(f"Skipping model due to evaluation error: {ve}")

    if not predictions:
        raise ValueError("No valid predictions were generated. Please check the models and data consistency.")

    # Pad predictions to have the same shape
    predictions = pad_to_max_shape(predictions)

    # Stack predictions for meta-model
    X_stack = np.vstack(predictions)  # Use vstack to ensure proper stacking of predictions

    # Ensure targets have the same length as the stacked predictions
    y = np.hstack(targets)  # Use hstack to flatten the targets

    # Check and adjust shapes to be consistent
    if len(y) != X_stack.shape[0]:
        raise ValueError(f"Shape mismatch: X_stack has {X_stack.shape[0]} samples, but y has {len(y)} samples.")

    # Ensure there are enough samples for train-test split
    if len(y) < 2:
        raise ValueError("Not enough samples to perform train-test split. Ensure the dataset is large enough.")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_stack, y, test_size=0.2, random_state=42)

    # Fit the meta-model
    meta_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = meta_model.predict(X_val)
    entity_scores = np.mean(predictions, axis=0)

    return {
        'entity_scores': entity_scores,
        'meta_model': meta_model,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'y_pred': y_pred
    }


def voting_ensemble(models: List[PyMADModel], data: Union[Dataset, Entity], eval_batch_size: int = 128) -> dict:
    """Evaluate a voting ensemble of models."""
    predictions = []
    for model in models:
        preds = evaluate_model(data, model, model_name=str(model), eval_batch_size=eval_batch_size)
        predictions.append(preds['entity_scores'])

    # Aggregate predictions (mean for soft voting)
    entity_scores = np.mean(predictions, axis=0)

    return {
        'entity_scores': entity_scores
    }


######################################################
# Genetic Algorithms
######################################################

def genetic_algorithm(models: List[PyMADModel], data: Union[Dataset, Entity], num_generations: int = 10,
                      population_size: int = 10) -> dict:
    """Genetic algorithm for model selection."""
    # Initial population
    population = [np.random.choice(models, size=len(models), replace=True) for _ in range(population_size)]

    for generation in range(num_generations):
        fitness_scores = []

        for individual in population:
            # Evaluate the individual (combination of models)
            ensemble_preds = bagging_ensemble(individual, data)
            fitness_score = np.mean(ensemble_preds['entity_scores'])
            fitness_scores.append(fitness_score)

        # Select the top individuals
        top_individuals = [population[idx] for idx in np.argsort(fitness_scores)[-population_size // 2:]]

        # Generate new population through crossover and mutation
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = np.random.choice(top_individuals, size=2, replace=False)
            crossover_point = np.random.randint(1, len(models) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

            new_population.append(child1)
            new_population.append(child2)

        # Mutation
        for individual in new_population:
            if np.random.rand() < 0.1:  # Mutation rate
                individual[np.random.randint(len(models))] = np.random.choice(models)

        population = new_population

    # Final evaluation of the best individual
    best_individual = population[np.argmax(fitness_scores)]
    ensemble_preds = bagging_ensemble(best_individual, data)

    return {
        'entity_scores': ensemble_preds['entity_scores']
    }

######################################################
# Helper prediction functions for predict
######################################################


def _predict_base(batch, model):
    Y, Y_hat, mask = model.forward(batch)
    if isinstance(Y, t.Tensor): Y = Y.detach().cpu().numpy()
    if isinstance(Y_hat, t.Tensor): Y_hat = Y_hat.detach().cpu().numpy()
    if isinstance(mask, t.Tensor): mask = mask.detach().cpu().numpy()
    Y_sigma = np.nan * np.ones(batch['Y'].shape)
    return Y, Y_hat, Y_sigma, mask


def _predict_dghl(batch, model):
    return _predict_base(batch, model)


def _predict_md(batch, model):
    return _predict_base(batch, model)


def _predict_nn(batch, model):
    return _predict_base(batch, model)


def _predict_rm(batch, model):
    return _predict_base(batch, model)


def _predict_lof(batch, model):
    return _predict_base(batch, model)


def _predict_kde(batch, model):
    return _predict_base(batch, model)


def _predict_abod(batch, model):
    return _predict_base(batch, model)

def _predict_sos(batch, model):
    return _predict_base(batch, model)

def _predict_alad(batch, model):
    return _predict_base(batch, model)

def _predict_pyod(batch, model):
    return _predict_base(batch, model)
def _predict_cof(batch, model):
    return _predict_base(batch, model)


def _predict_cblof(batch, model):
    return _predict_base(batch, model)


def _predict_rnn(batch, model):
    batch_size, n_features, window_size = batch['Y'].shape
    Y, Y_hat, mask = model.forward(batch)
    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(
    ), mask.detach().cpu().numpy()
    Y = Y.reshape(n_features, -1)[:, :window_size]  # to [n_features, n_time]
    Y_hat = Y_hat.reshape(n_features,
                          -1)[:, :window_size]  # to [n_features, n_time]
    mask = mask.reshape(n_features,
                        -1)[:, :window_size]  # to [n_features, n_time]

    # Add mask dimension
    Y = Y[None, :, :]
    Y_hat = Y_hat[None, :, :]
    mask = mask[None, :, :]

    Y_sigma = np.nan * np.ones(batch['Y'].shape)
    return Y, Y_hat, Y_sigma, mask


def _predict_lstmvae(batch, model):
    Y, Y_mu, mask, Y_sigma, *_ = model.forward(batch)
    Y, Y_hat, mask, Y_sigma = Y.detach().cpu().numpy(), Y_mu.detach().cpu(
    ).numpy(), mask.detach().cpu().numpy(), Y_sigma.detach().cpu().numpy()
    return Y, Y_hat, Y_sigma, mask

def get_eval_batchsizes(model_name: str) -> int:
    """Return evaluation batch sizes of algorithm
    """
    _VALID_MODEL_NAMES = ['RNN', 'DGHL', 'LSTMVAE', 'MD', 'RM',
                          'NN', 'LOF','ABOD','KDE','COF','CBLOF','SOS']  # TODO: Should be stored somewhere centrally

    model_type = model_name.split('_')[0]

    if model_type == 'RNN':
        return RNN_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'DGHL':
        return DGHL_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'NN':
        return NN_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'MD':
        return MD_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'RM':
        return RM_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'LSTMVAE':
        return LSTMVAE_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'LOF':
        return LOF_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'ABOD':
        return ABOD_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'KDE':
        return KDE_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'CBLOF':
        return CBLOF_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'COF':
        return COF_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'SOS':
        return SOS_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'ALAD':
        return ALAD_TRAIN_PARAM_GRID['eval_batch_size'][0]
    else:
        return PYOD_TRAIN_PARAM_GRID['eval_batch_size'][0]


def _adjust_scores_with_padding(scores: np.ndarray,
                                padding_size: int = 0,
                                padding_type: str = 'right'):
    if scores.ndim == 1: scores = scores[None, :]

    if (padding_type == 'right') and (padding_size > 0):
        scores = scores[:, :-padding_size]
    elif (padding_type == 'left') and (padding_size > 0):
        scores = scores[:, padding_size:]
    return scores
