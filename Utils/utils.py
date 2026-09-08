import numpy as np
import torch as t
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from Utils.config import Config
from Utils.pipeline_spec import (parse_anomaly_rate, parse_anomaly_type,
                                 parse_decision_metrics, parse_detectors,
                                 parse_stages)
from pathlib import Path
import os
from loguru import logger

from PIL import Image
from io import BytesIO
import base64
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
path = os.path.join(BASE_DIR, '')
config_path = os.path.join(path, 'Configs/config.yml')
logger.info(f'{__file__}\ncurrent base dir is path is {path}\nconfig_path path is {config_path}')





def get_args_from_cmdline():
    """
    Get arguments from command line. Useful for 
    experiments.
    """
    parser = ArgumentParser(description='Config file')
    parser.add_argument('--config_file_path',
                        '-c', 
                        type=str, 
                        default=config_path,
                        required=False,
                        help='path to config file')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='Dataset name (e.g., skab, smd)')
    parser.add_argument('--entity',
                        type=str,
                        default=None,
                        help='Entity ID (e.g., 3, 5)')
    parser.add_argument('--parallel',
                        type=str,
                        default='false',
                        help='Run model selection in parallel (true/false, t/f, T/F, True/False)')
    parser.add_argument('--enable_online',
                        action='store_true',
                        help='Enable online phase processing (flag, no value needed)')
    parser.add_argument('--update_interval',
                        type=int,
                        default=5,
                        help='Number of windows between model selection re-optimization (default: 5)')
    parser.add_argument('--iteration',
                        type=int,
                        default=5,
                        help='Number of iterations for window sizing (window_size = online_data / iteration, default: 5)')
    parser.add_argument('--strategy',
                        type=str,
                        default='adaptive',
                        choices=['adaptive', 'fixed-best', 'fixed-random'],
                        help='Online adaptation strategy: adaptive (re-optimize), fixed-best (no reopt, use best offline), fixed-random (no reopt, random model)')
    parser.add_argument('--inject_online_regime',
                        action='store_true',
                        help='Inject regime shifts (scale+wander) into online data only (flag, no value needed)')
    parser.add_argument('--max_online_windows',
                        type=int,
                        default=None,
                        help='Maximum number of online windows to process (default: None = all windows)')
    parser.add_argument('--skip_gan',
                        action='store_true',
                        help='Skip GAN robustness testing for faster execution (testing/debugging)')
    parser.add_argument('--explain',
                        action='store_true',
                        help='Enable all explainability outputs (reports/plots). Explainability is OFF by default.')
    parser.add_argument('--llm_model',
                        type=str,
                        default=None,
                        help='Model for the automatic LLM narration after an --explain run '
                             '(default: qwen2.5:14b-instruct). Narration is skipped with a '
                             'warning if no local LLM server is reachable.')
    parser.add_argument('--llm_base_url',
                        type=str,
                        default=None,
                        help='OpenAI-compatible base URL for LLM narration '
                             '(default: http://localhost:11434/v1, i.e. Ollama).')
    parser.add_argument('--stages',
                        type=str,
                        default='all',
                        help="Comma-separated sub-stages of the model-selection pipeline to run. "
                             "Tokens: ga, thompson, gan, offby, montecarlo (plus 'all' and "
                             "'robustness'=gan,offby,montecarlo). Default 'all' runs the full pipeline; "
                             "any strict subset runs only those stages + their explainability, then stops "
                             "(no rank aggregation / final decision / online phase) and runs sequentially.")

    parser.add_argument('--anomaly_type',
                        type=str,
                        default=None,
                        help="Synthetic anomaly injected at stage 4. One of: "
                             "spikes, contextual, flip, speedup, noise, cutoff, scale, "
                             "wander, average. Default 'spikes'.")
    parser.add_argument('--anomaly_rate',
                        type=str,
                        default=None,
                        help="Target fraction of timesteps to label anomalous, in (0, 1]. "
                             "For 'spikes' this is the per-timestep injection probability; "
                             "for the other types it sizes the injected segment. Omit to "
                             "keep the per-type defaults.")

    parser.add_argument('--decision_metric',
                        type=str,
                        default=None,
                        help="Comma-separated metrics the fitness function is built "
                             "from: f1, pr_auc (both by default), vus. Each may carry a weight "
                             "('f1:0.5,pr_auc:0.3,vus:0.2'); unweighted metrics count "
                             "equally and weights are normalised. The fitness is their "
                             "weighted mean, and it is what the GA, Thompson Sampling "
                             "and the final ensemble-vs-single comparison all maximise.")

    parser.add_argument('--overwrite',
                        type=str,
                        default=None,
                        choices=['true', 'false'],
                        help="Retrain the base detectors even when checkpoints already exist. "
                             "Overrides `overwrite` in the config file. Training dominates the "
                             "runtime, so 'false' reuses existing models and is much faster.")

    parser.add_argument('--detectors',
                        type=str,
                        default=None,
                        help="Comma-separated base detectors to select among, e.g. "
                             "'LOF_1,NN_2,CBLOF_3'. Default: all 11. Only the families of the "
                             "requested detectors are trained. Detectors with no trained model "
                             "for the chosen dataset/entity are skipped with a warning; at least "
                             "two must remain.")

    cmd_args = parser.parse_args()
    
    # Load config from file
    if cmd_args.config_file_path and os.path.exists(cmd_args.config_file_path):
        config_file = cmd_args.config_file_path
    else:
        # Fallback to default config path (dynamically determined from BASE_DIR)
        config_file = config_path
    
    args = Config(config_file_path=config_file).parse()
    
    # Override with command line arguments if provided
    if cmd_args.dataset is not None:
        args['dataset'] = cmd_args.dataset
    if cmd_args.entity is not None:
        args['entity'] = cmd_args.entity
    
    # Parse parallel flag (accepts: true, t, T, True, TRUE, 1, yes, y, Y, Yes, YES)
    parallel_str = cmd_args.parallel.lower()
    args['parallel'] = parallel_str in ['true', 't', '1', 'yes', 'y']
    
    # Parse enable_online flag (now it's a boolean action flag)
    args['enable_online'] = cmd_args.enable_online  # Already boolean from action='store_true'
    
    # Update interval from command line
    args['update_interval'] = cmd_args.update_interval
    
    # Iteration parameter for window sizing
    args['iteration'] = cmd_args.iteration
    
    # Strategy for online adaptation
    args['strategy'] = cmd_args.strategy
    
    # Online regime shift injection flag
    args['inject_online_regime'] = cmd_args.inject_online_regime
    
    # Max online windows parameter
    args['max_online_windows'] = cmd_args.max_online_windows
    
    # Skip GAN flag
    args['skip_gan'] = cmd_args.skip_gan

    # Explainability is OFF by default; --explain enables it everywhere.
    args['explain'] = cmd_args.explain
    # LLM narration overrides (None → Explainability.llm defaults in app.py).
    args['llm_model'] = cmd_args.llm_model
    args['llm_base_url'] = cmd_args.llm_base_url

    # Which pipeline sub-stages to run, and which detectors to select among.
    # Both vocabularies live in Utils/pipeline_spec.py so app.py and the web UI
    # read the same definitions ('all' and 'robustness' are stage groups).
    try:
        args['stages'] = parse_stages(cmd_args.stages)
    except ValueError as e:
        parser.error(str(e))
    try:
        args['detectors'] = parse_detectors(cmd_args.detectors)
    except ValueError as e:
        parser.error(str(e))
    try:
        args['anomaly_type'] = parse_anomaly_type(cmd_args.anomaly_type)
        args['anomaly_rate'] = parse_anomaly_rate(cmd_args.anomaly_rate)
        args['decision_metrics'] = parse_decision_metrics(cmd_args.decision_metric)
    except ValueError as e:
        parser.error(str(e))

    # CLI overrides the config file, so a caller can reuse checkpoints without
    # editing Configs/config.yml.
    if cmd_args.overwrite is not None:
        args['overwrite'] = cmd_args.overwrite == 'true'

    return args

def de_unfold(windows, window_step):
    """
    windows of shape (n_windows, n_channels, window_size)
    """
    n_windows, n_channels, window_size = windows.shape

    if window_step < 0:
        window_step = window_size

    assert window_step <= window_size, 'Window step must be smaller than window_size'

    total_len = (n_windows)*window_step + (window_size-window_step)

    x = t.zeros((n_channels, total_len))
    counter = t.zeros((1, total_len))

    for i in range(n_windows):
        start = i*window_step
        end = start + window_size
        x[:, start:end] += windows[i]
        counter[:, start:end] += 1

    x = x/counter

    return x

def visualize_predictions(predictions: dict, savefig=True):
    """Visualizes univariate algorithm given the predictions dictionary
    """
    MODEL_NAMES = list(predictions.keys())
    fig, axes = plt.subplots(len(MODEL_NAMES),
                             1,
                             sharey=True,
                             sharex=True,
                             figsize=(30, 5 * len(MODEL_NAMES)))

    for i, model_name in enumerate(MODEL_NAMES):
        start_anomaly = np.argmax(
            np.diff(predictions[model_name]['anomaly_labels'].flatten()))
        end_anomaly = np.argmin(
            np.diff(predictions[model_name]['anomaly_labels'].flatten()))
        axes[i].plot(predictions[model_name]['Y'].flatten(),
                     color='darkblue',
                     label='Y')
        axes[i].plot(predictions[model_name]['Y_hat'].flatten(),
                     color='darkgreen',
                     label='Y_hat')
        axes[i].plot(
            np.arange(start_anomaly, end_anomaly),
            predictions[model_name]['Y'].flatten()[start_anomaly:end_anomaly],
            color='red',
            label='Anomaly')

        entity_scores = predictions[model_name]['entity_scores'].flatten()
        entity_scores = (entity_scores - entity_scores.min()) / (
            entity_scores.max() - entity_scores.min())
        # entity_scores = (entity_scores - entity_scores.mean())/(entity_scores.std())
        axes[i].plot(entity_scores,
                     color='magenta',
                     linestyle='--',
                     label='Anomaly Scores')

        axes[i].set_title(f'Predictions of Model {model_name}', fontsize=16)
        axes[i].legend(fontsize=16, ncol=2, shadow=True, fancybox=True)
        axes[i].set_xlabel('Time', fontsize=16)
        axes[i].set_ylabel('Y', fontsize=16)

    if savefig:
        plt.savefig('predictions.pdf')
    # plt.show()


def visualize_data(train_data, test_data, savefig=False,save_path=None):
    """Visualizes train and testing splits of a univariate entity.
    """
    # Visualize the train and the test data
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 4))
    axes[0].plot(train_data.entities[0].Y.flatten(), color='darkblue')
    axes[0].set_title('Train data', fontsize=16)

    start_anomaly = np.argmax(np.diff(test_data.entities[0].labels.flatten()))
    end_anomaly = np.argmin(np.diff(test_data.entities[0].labels.flatten()))

    axes[1].plot(test_data.entities[0].Y.flatten(),
                 color='darkblue',
                 label='Y')
    axes[1].plot(np.arange(start_anomaly, end_anomaly),
                 test_data.entities[0].Y.flatten()[start_anomaly:end_anomaly],
                 color='red',
                 label='Anomaly')
    axes[1].set_title('Test data', fontsize=16)
    axes[1].legend(fontsize=16, ncol=2, shadow=True, fancybox=True)

    if savefig:
        plt.savefig(save_path,format = 'png')
        plt.clf()
        plt.close()
    else:
        # plt.show()
        print('Visualization of train and test data is done!')



def img2binary(img_path:Optional[str]=None)->str:
    save_file = BytesIO()
    load_img = Image.open(img_path)
    load_img.save(save_file, format='png')
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf-8')
    return save_file_base64