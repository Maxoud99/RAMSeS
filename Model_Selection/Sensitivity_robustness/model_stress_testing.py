import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli, norm
from sklearn.neighbors import NearestNeighbors
from scipy.signal import fftconvolve, find_peaks
from scipy import interpolate
from sklearn.model_selection import ParameterGrid
from loguru import logger
from Metrics.metrics import range_based_precision_recall_f1_auc, f1_soft_score
from Utils.model_selection_utils import evaluate_model
from Model_Selection.inject_anomalies import InjectAnomalies

ANOMALY_PARAM_GRID = {
    'spikes': {
        'anomaly_type': ['spikes'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2, 5, 10, 20],
        'anomaly_propensity': [0.5],
    },
    'contextual': {
        'anomaly_type': ['contextual'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'flip': {
        'anomaly_type': ['flip'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [0.01, .6, 2, 20],
    },
    'speedup': {
        'anomaly_type': ['speedup'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'speed': [0.25, 0.5, 2, 4],
        'scale': [2],
    },
    'noise': {
        'anomaly_type': ['noise'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'noise_std': [0.05],
        'scale': [2],
    },
    'cutoff': {
        'anomaly_type': ['cutoff'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'constant_type': ['noisy_0', 'noisy_1'],
        'constant_quantile': [0.75],
        'scale': [2],
    },
    'scale': {
        'anomaly_type': ['scale'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'amplitude_scaling': [2, 5, 10, 20],
        'scale': [2],
    },
    'wander': {
        'anomaly_type': ['wander'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'baseline': [-0.3, -0.1, 0.1, 0.3],
        'scale': [2],
    },
    'average': {
        'anomaly_type': ['average'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'ma_window': [4, 8],
        'scale': [2],
    }
}


def Inject(data, anomaly_types):
    random_state = np.random.randint(1, 10000)

    anomaly_obj = InjectAnomalies(random_state=random_state,
                                  verbose=False,
                                  max_window_size=128,
                                  min_window_size=8)
    T = data.entities[0].Y
    data_std = max(np.std(T), 0.01)
    T_a_concatenated = []
    anomaly_sizes_concatenated = []
    anomaly_labels_concatenated = []

    for anomaly in anomaly_types:
        for anomaly_params in list(
                ParameterGrid(ANOMALY_PARAM_GRID[anomaly])):
            anomaly_params['T'] = T
            anomaly_params['scale'] = anomaly_params['scale'] * data_std

            # Inject synthetic anomalies to the data
            T_a, anomaly_sizes, anomaly_labels = anomaly_obj.inject_anomalies(
                **anomaly_params)
            anomaly_sizes = anomaly_sizes / data_std

            T_a_concatenated.append(T_a)
            anomaly_sizes_concatenated.append(anomaly_sizes)
            anomaly_labels_concatenated.append(anomaly_labels)

    T_a_concatenated = np.concatenate(T_a_concatenated, axis=1)
    anomaly_sizes_concatenated = np.concatenate(anomaly_sizes_concatenated,
                                                axis=0)
    anomaly_labels_concatenated = np.concatenate(
        anomaly_labels_concatenated, axis=0)

    data.entities[0].Y = T_a_concatenated
    data.entities[0].n_time = T_a_concatenated.shape[1]
    data.entities[0].mask = np.ones((T_a_concatenated.shape))
    data.entities[0].labels = anomaly_labels_concatenated
    data.total_time = T_a_concatenated.shape[1]
    return data, anomaly_sizes


def run_robustness_tests(data, anomaly_types, model_names, trained_models):
    results = {}

    for anomaly_type in anomaly_types:
        modified_data, _ = Inject(data, [anomaly_type])
        anomaly_start = np.argmax(modified_data.entities[0].labels)
        anomaly_end = modified_data.entities[0].Y.shape[1] - np.argmax(modified_data.entities[0].labels[::-1])

        # Visualization setup
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
        axes[0].plot(modified_data.entities[0].Y.flatten(), color='darkblue')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     modified_data.entities[0].Y.flatten()[anomaly_start:anomaly_end], color='red')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     data.entities[0].Y.flatten()[anomaly_start:anomaly_end], color='darkblue', linestyle='--')
        axes[0].set_title('Test Data with Injected Anomalies', fontsize=16)
        axes[1].plot(_.flatten(), color='pink')
        axes[1].plot(modified_data.entities[0].labels.flatten(), color='red')
        axes[1].set_title('Anomaly Scores', fontsize=16)
        # plt.show()

        # Evaluating each model on the modified data
        results[anomaly_type] = []
        for model_name in model_names:
            model = trained_models.get(model_name)
            if model:
                evaluation = evaluate_model(modified_data, model, model_name)  # Assume this function returns a dict
                y_true = evaluation['anomaly_labels'].flatten()
                y_scores = evaluation['entity_scores'].flatten()
                _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true, y_scores)
                results[anomaly_type].append({'model': model_name, 'f1': best_f1, 'pr_auc': pr_auc})
                logger.info(f"Evaluated {model_name} on {anomaly_type}: F1={best_f1}, PR_AUC={pr_auc}")

        # Sorting results within this anomaly type by F1 and PR-AUC
        results[anomaly_type].sort(key=lambda x: x['f1'], reverse=True)
        logger.info(f"Models ranked by F1 for {anomaly_type}: {[x['model'] for x in results[anomaly_type]]}")
        results[anomaly_type].sort(key=lambda x: x['pr_auc'], reverse=True)
        logger.info(f"Models ranked by PR-AUC for {anomaly_type}: {[x['model'] for x in results[anomaly_type]]}")

    return results

