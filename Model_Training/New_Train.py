# time_series_framework/Model_Training/New_Train.py
from sklearn.model_selection import ParameterGrid
from loguru import logger
from Loaders.loader import Loader
import matplotlib.pyplot as plt
from Algorithms.lof import TsadLof
import os

class TrainModels:
    def __init__(self, dataset, entity, algorithm_list, downsampling, min_length, root_dir,
                 training_size, overwrite, verbose, save_dir):
        self.dataset = dataset
        self.entity = entity
        self.algorithm_list = algorithm_list
        self.downsampling = downsampling
        self.min_length = min_length
        self.root_dir = root_dir
        self.training_size = training_size
        self.overwrite = overwrite
        self.verbose = verbose
        self.save_dir = save_dir
        self.train_data = None
        self.test_data = None

    def train_lof(self, model_hyper_params, train_hyper_params, model_id):
        model = TsadLof(**model_hyper_params)

        if not self.overwrite and self.logging_obj.check_file_exists(obj_class=self.logging_hierarchy, obj_name=f"LOF_{model_id + 1}"):
            print(f'Model LOF_{model_id + 1} already trained!')
            return

        dataloader = Loader(
            dataset=self.train_data,
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
        model.fit(dataloader)

        img_name = f"LOF_{model_id + 1}.png"
        img_path = os.path.join(self.img_dir, img_name)
        logger.info(f'img_path is {img_path}')

        test_dataloader = Loader(
            dataset=self.test_data,
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

        for batch in test_dataloader:
            Y, Y_hat, _ = model.forward(batch)
            Y, Y_hat = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy()
            # Visualization and saving logic

        self.logging_obj.save(obj=model, obj_name=f"LOF_{model_id}", obj_meta={
            'train_hyperparameters': train_hyper_params,
            'model_hyperparameters': model_hyper_params
        }, obj_class=self.logging_hierarchy)
