# import matplotlib.pyplot as plt
# import os
# import logging
# import torch as t
# from Datasets.load import load_data
# from Utils.utils import get_args_from_cmdline
# from Model_Training.train import TrainModels
# from loguru import logger
# import traceback
# import numpy as np
# from Model_Selection.inject_anomalies import Inject
# from Metrics.Ensemble_GA import genetic_algorithm, evaluate_individual_models, fitness_function
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
#
# save_dir = "Mononito/trained_models/anomaly_archive/006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2/"
#
# args = get_args_from_cmdline()
# data_dir = args['dataset_path']
# train_data = load_data(dataset='anomaly_archive', group='train',
#                        entities='006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2', downsampling=10,
#                        min_length=256, root_dir=data_dir, normalize=True, verbose=False)
# test_data = load_data(dataset='anomaly_archive', group='test',
#                       entities='006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2', downsampling=10,
#                       min_length=256, root_dir=data_dir, normalize=True, verbose=False)
# test_data, anomaly_sizes = Inject(test_data, ['spikes'])
# data = test_data.entities[0].Y
# targets = test_data.entities[0].labels
# print(type(data))
# print(type(targets))
#
# # Plot the data
# fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
#
# # First row: plot the data
# axes[0].plot(data.flatten(), label='Data', color='blue')
# axes[0].set_title('Data')
# axes[0].set_ylabel('Value')
# axes[0].legend()
# axes[0].grid(True)
#
# # Second row: plot the labels with spikes
# axes[1].plot(targets, label='Labels', color='gray')
# spike_indices = np.where(targets.flatten() == 1)[0]
# spike_values = np.ones_like(spike_indices)  # Set spikes at 1 for visibility
# axes[1].vlines(spike_indices, ymin=0, ymax=spike_values, color='red', label='Anomalies')
# axes[1].set_title('Labels')
# axes[1].set_xlabel('Time (index)')
# axes[1].set_ylabel('Label')
# axes[1].grid(True)
#
# plt.tight_layout()
# # plt.show()