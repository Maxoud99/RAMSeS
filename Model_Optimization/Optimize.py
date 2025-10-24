# time_series_framework/Model_Optimization/Optimize.py
import os
import numpy as np
import torch as t
from sklearn.metrics import mean_squared_error
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from Loaders.loader import Loader
from Algorithms.lof import TsadLof

# Set a shorter temporary directory for Ray Tune
os.environ["TUNE_RESULT_DIR"] = "D:/ray_results"

class DataLoaderWrapper:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.Y_windows = t.cat([batch['Y'].to(device) for batch in dataloader])
        self.X_windows = t.cat([batch['X'].to(device) for batch in dataloader]) if dataloader.X_windows is not None else None
        self.mask_windows = t.cat([batch['mask'].to(device) for batch in dataloader])

def convert_to_tensor(batch, device):
    return {k: (t.tensor(v).to(device) if isinstance(v, np.ndarray) else v.to(device)) for k, v in batch.items()}

def train_lof(config, train_data, test_data):
    device = "cuda" if t.cuda.is_available() else "cpu"

    model_hyper_params = {
        'window_size': int(config['window_size']),
        'window_step': int(config['window_step']),
        'device': device
    }

    train_hyper_params = {
        'batch_size': int(config['batch_size'])
    }

    model = TsadLof(**model_hyper_params).to(device)  # Move model to GPU
    train_loader = Loader(
        dataset=train_data,
        batch_size=train_hyper_params['batch_size'],
        window_size=model_hyper_params['window_size'],
        window_step=model_hyper_params['window_step'],
        shuffle=False,
        padding_type='right',
        sample_with_replace=False,
        verbose=False,
        mask_position='None',
        n_masked_timesteps=0
    )

    wrapped_train_loader = DataLoaderWrapper(train_loader, device)

    for batch in wrapped_train_loader.dataloader:
        batch = convert_to_tensor(batch, device)  # Convert batch data to tensors and move to GPU
        model.fit(wrapped_train_loader)

    test_loader = Loader(
        dataset=test_data,
        batch_size=train_hyper_params['batch_size'],
        window_size=model_hyper_params['window_size'],
        window_step=model_hyper_params['window_step'],
        shuffle=False,
        padding_type='right',
        sample_with_replace=False,
        verbose=False,
        mask_position='None',
        n_masked_timesteps=0
    )

    wrapped_test_loader = DataLoaderWrapper(test_loader, device)

    Y_true, Y_pred = [], []
    for batch in wrapped_test_loader.dataloader:
        batch = convert_to_tensor(batch, device)  # Convert batch data to tensors and move to GPU
        Y, Y_hat, _ = model.forward(batch)
        Y, Y_hat = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy()
        Y_true.append(Y)
        Y_pred.append(Y_hat)

    Y_true = np.concatenate(Y_true).flatten()
    Y_pred = np.concatenate(Y_pred).flatten()

    mse = mean_squared_error(Y_true, Y_pred)
    tune.report(mean_squared_error=mse)

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id}"

def run_optimization(train_data, test_data):
    config = {
        'window_size': tune.quniform(50, 80, 1),
        'window_step': tune.quniform(1, 10, 1),
        'batch_size': tune.quniform(100, 140, 1)
    }

    scheduler = ASHAScheduler(
        metric="mean_squared_error",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["mean_squared_error", "training_iteration"]
    )

    result = tune.run(
        tune.with_parameters(train_lof, train_data=train_data, test_data=test_data),
        resources_per_trial={"cpu": 4, "gpu": 1},  # Allocate one GPU per trial
        config=config,
        num_samples=5,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path="D:/ray_results",  # Use the custom directory for Ray Tune results
        trial_dirname_creator=trial_dirname_creator  # Custom trial directory names
    )

    best_trial = result.get_best_trial("mean_squared_error", "min", "last")
    best_params = best_trial.config
    return best_params
