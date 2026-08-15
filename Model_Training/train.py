#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# Script to train algorithm on a dataset of entities
#######################################
import json
from typing import List, Union, Optional
from sklearn.model_selection import ParameterGrid
import os
from tqdm import tqdm
from loguru import logger
import numpy as np
import torch as t
from Algorithms.alad import TsadALAD
from Algorithms.pyod_model import PyodModel
from Algorithms.tsbad_model import TSBADModel
from .hyperparameter_grids import *  # DGHL_TRAIN_PARAM_GRID, DGHL_PARAM_GRID, MD_TRAIN_PARAM_GRID, MD_PARAM_GRID, RM_PARAM_GRID, RM_TRAIN_PARAM_GRID, NN_PARAM_GRID, NN_TRAIN_PARAM_GRID, LSTMVAE_TRAIN_PARAM_GRID, LSTMVAE_PARAM_GRID, RNN_TRAIN_PARAM_GRID, RNN_PARAM_GRID
from Model_Training.training_args import TrainingArguments
from Model_Training.trainer import Trainer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from Utils.logger import Logger
from Utils.pipeline_spec import TRANSDUCTIVE_FAMILIES, TSBAD_FAMILIES
from Datasets.load import load_data
from Loaders.loader import Loader
# Import all the algorithm here!
from Algorithms.dghl import DGHL
from Algorithms.rnn import RNN
from Algorithms.lstmvae import LSTMVAE
from Algorithms.nearest_neighbors import NearestNeighbors
from Algorithms.mean_deviation import MeanDeviation
from Algorithms.running_mean import RunningMean
from Algorithms.lof import TsadLof
from Algorithms.kde import TsadKde
from Algorithms.abod import TsadABOD
from Algorithms.cblof import TsadCblof
from Algorithms.cof import  TsadCof
from Algorithms.sos import TsadSOS

class TrainModels(object):
    """Class to pre-train algorithm on a dataset/entity.
    
    Parameters
    ----------
    dataset: str
        Name of dataset in which the entity belongs. 
    entity: str
        Multivariate timeseries entity on which we need to evaluate performance of algorithm.

    downsampling: int

    root_dir: str
        Dataset directory
    batch_size: int 
        Batch size for evaluation
    training_size: float
        Percentage of training data to use for training algorithm.
    overwrite: bool
        Whether to re-train existing algorithm.
    verbose: bool
        Controls verbosity
    save_dir: str
        Directory to save the trained algorithm.
    """

    def __init__(self,
                 dataset:str='anomaly_archive',
                 entity:str='233_UCR_Anomaly_mit14157longtermecg',
                 algorithm_list:list[str]=None,
                 downsampling:Optional[int]=None,
                 min_length:Optional[int]=None,
                 root_dir:str='../../datasets/',
                 training_size:float=1,
                 overwrite:bool = False,
                 verbose:bool = True,
                 save_dir:str='Mononito/trained_models'):

        if training_size > 1.0:
            raise ValueError('Training size must be <= 1.0')
        self.save_dir = save_dir
        self.overwrite = overwrite  # Store overwrite flag

        self.img_dir = os.path.join(self.save_dir, f"{dataset}/{entity}")
        logger.info(f'self.img_dir is {self.img_dir}')
        self.verbose = verbose


        self.train_data = load_data(dataset=dataset,
                                    group='train',
                                    entities=entity,
                                    downsampling=downsampling,
                                    min_length=min_length,
                                    root_dir=root_dir,
                                    verbose=False)

        self.test_data = load_data(dataset=dataset,
                                    group='test',
                                    entities=entity,
                                    downsampling=downsampling,
                                    min_length=min_length,
                                    root_dir=root_dir,
                                    verbose=False)

        if verbose:
            print(f'Number of entities: {self.train_data.n_entities}')
            print(
                f'Using the first {training_size*100}\% of the training data to train the algorithm.'
            )

        self.overwrite = overwrite

        for e_i in range(self.train_data.n_entities):
            t = self.train_data.entities[e_i].Y.shape[1]
            self.train_data.entities[e_i].Y = self.train_data.entities[
                e_i].Y[:, :int(training_size * t)]

        # Logger object to save the algorithm
        self.logging_obj = Logger(save_dir=self.save_dir,
                                  overwrite=self.overwrite,
                                  verbose=verbose)
        self.logging_hierarchy = [dataset, entity]

        # already selected algorithm
        self._VALID_MODEL_ARCHITECTURES = algorithm_list
        self.batch_size=8

    def _diagnostic_batch_size(self, family: str, window_size: int) -> int:
        """Batch size for the post-fit `.png` loop, which runs `model.forward`.

        `self.batch_size` is 8, and for the transductive families that is not
        merely suboptimal but fatal: COF raises IndexError on any call holding
        fewer rows than its `n_neighbors` (20), and SpectralResidual raises on a
        batch of exactly 2 and returns three scores for a batch of 1. The loop
        runs BEFORE `logging_obj.save`, so the raise means no checkpoint is
        written at all — these families could not be trained.

        The TSB-AD families need it for the same practical reason and a
        different cause: each cuts `detector__win_size` subsequences (30-100
        timesteps) out of whatever call it is given, and eight rows contain no
        such subsequence.

        Handing them the whole series in one batch fixes it and matches how
        `Utils.model_selection_utils.evaluate_model` scores them, so the picture
        the plot shows is the scoring the pipeline will actually do. Same `+
        window_size` slack as there, for the loader's right-padding.
        """
        if family.upper() not in TRANSDUCTIVE_FAMILIES | TSBAD_FAMILIES:
            return self.batch_size
        n_time = max(e.Y.shape[1] for e in self.test_data.entities)
        return max(1, n_time + max(1, int(window_size)))

    def train_models(self, model_architectures: List[str] = 'all'):
        """Function to selected algorithm.
        """


        model_architectures = self._VALID_MODEL_ARCHITECTURES
        logger.info(f'model_architectures is {model_architectures}')



        os.path.exists(self.img_dir) or os.makedirs(self.img_dir)
        vis_data_path = os.path.join(self.img_dir,'data.png')
        logger.info(f'vis_data_path is {vis_data_path}')
        self.visualization_data(_vis_data_path=vis_data_path)



        # No family-level skip list any more.
        #
        # This used to read the .png files in the directory, take everything
        # before the first underscore, and skip that whole family. Two problems.
        # It was a DIFFERENT signal from the one everything else uses — the web
        # UI, and `Logger.check_file_exists` below, both read `.pth` — so a
        # detector could be reported as untrained and then skipped by the
        # trainer. And it was all-or-nothing per family: one LOF checkpoint on
        # disk meant LOF_2..LOF_4 were never trained, which is the state five of
        # the entities here are actually in.
        #
        # Every train_* method already checks `check_file_exists` per instance
        # before fitting, on `.pth`, and honours `self.overwrite` while doing it.
        # So the coarse gate was redundant with a correct check that sits one
        # level down; dropping it lets a partially trained family fill itself in,
        # and costs an already-complete family only the call that then skips
        # every instance.
        if self.overwrite:
            logger.info('Overwrite=True: Retraining all models regardless of existing files')
        exist_model_list = []

        for model_name in model_architectures:
            # if no cache train model
            if ('DGHL' == model_name) & (model_name not in exist_model_list):
                self.train_dghl()
            elif ('RNN' == model_name) & (model_name not in exist_model_list) :
                self.train_rnn()
            elif ('LSTMVAE' == model_name) & (model_name not in exist_model_list):
                self.train_lstmvae()
            elif ('NN' == model_name) & (model_name not in exist_model_list):
                self.train_nn()
            elif ('MD' == model_name) & (model_name not in exist_model_list):
                self.train_md()
            elif ('RM' == model_name) & (model_name not in exist_model_list):
                self.train_rm()
            elif ('LOF' == model_name) & (model_name not in exist_model_list):

                self.train_lof()

            elif ('KDE' == model_name) & (model_name not in exist_model_list):
                self.train_kde()

            elif ('ABOD' == model_name) & (model_name not in exist_model_list):
                self.train_abod()

            elif ('CBLOF' == model_name) & (model_name not in exist_model_list):
                self.train_cblof()

            elif ('COF' == model_name) & (model_name not in exist_model_list):
                self.train_cof()

            elif ('SOS' == model_name) & (model_name not in exist_model_list):
                self.train_sos()
            elif ('ALAD' == model_name) & (model_name not in exist_model_list):
                self.train_alad()
            elif (model_name.upper() in TSBAD_FAMILIES) & (model_name not in exist_model_list):
                self.train_tsbad(model_name)
            elif (model_name not in exist_model_list):
                self.train_pyod(model_name)






    def visualization_data(self,_vis_data_path):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 4))
        axes[0].plot(self.train_data.entities[0].Y.flatten())
        axes[0].set_title('Train data')
        axes[1].plot(self.test_data.entities[0].Y.flatten())
        axes[1].set_title('Test data')
        plt.savefig(_vis_data_path,format='png')
        plt.clf()
        plt.close()


    def train_lof(self,batch_size=32):

        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(LOF_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(LOF_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadLof(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"LOF_{MODEL_ID + 1}"):
                        print(f'Model LOF_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"LOF_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
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

                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()



                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"LOF_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_kde(self, batch_size=32):

        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(KDE_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(KDE_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadKde(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"KDE_{MODEL_ID + 1}"):
                        print(f'Model KDE_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"KDE_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"KDE_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)
    def train_abod(self, batch_size=32):

        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(ABOD_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(ABOD_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadABOD(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"ABOD_{MODEL_ID + 1}"):
                        print(f'Model ABOD_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"ABOD_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"ABOD_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_cblof(self, batch_size=32):

        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(CBLOF_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(CBLOF_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadCblof(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"CBLOF_{MODEL_ID + 1}"):
                        print(f'Model CBLOF_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"CBLOF_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"CBLOF_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)
    def train_cof(self, batch_size=32):

        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(COF_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(COF_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadCof(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"COF_{MODEL_ID + 1}"):
                        print(f'Model COF_{MODEL_ID + 1} already trained!')
                        print("Params")
                        # print(model_hyper_params)
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"COF_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self._diagnostic_batch_size(
                        'COF', model_hyper_params['window_size']),
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"COF_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)
    def train_dghl(self):


        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(DGHL_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(DGHL_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                if self.train_data.entities[
                        0].X is not None:  # DGHL also considers covariates
                    model_hyper_params[
                        'n_features'] = self.train_data.n_features + self.train_data.entities[
                            0].X.shape[0]
                else:
                    model_hyper_params[
                        'n_features'] = self.train_data.n_features

                training_args = TrainingArguments(**train_hyper_params)
                model = DGHL(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"DGHL_{MODEL_ID+1}"):
                        print(f'Model DGHL_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1




                # eval the model
                trainer.model.eval()
                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0
                )

                img_name = f"DGHL_{MODEL_ID}.png"
                img_path = os.path.join(self.img_dir,img_name)
                logger.info(f'img_path is {img_path} ')

                # Declare the eval data loader
                for batch in test_dataloader:
                    Y, Y_hat, mask = trainer.model.forward(batch)
                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()
                    break

                batch_num = 0  # Use first batch (safe for any batch size)
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()



                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"DGHL_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_md(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(MD_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(MD_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model_hyper_params['n_features'] = self.train_data.n_features
                training_args = TrainingArguments(**train_hyper_params)
                model = MeanDeviation(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"MD_{MODEL_ID+1}"):
                        print(f'Model MD_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1

                # eval the model
                trainer.model.eval()
                test_dataloader = Loader(dataset=self.test_data,
                                    batch_size=8,
                                    window_size=64,
                                    window_step=64,
                                    shuffle=False,
                                    padding_type='right',
                                    sample_with_replace=False,
                                    verbose=False,
                                    mask_position='None',
                                    n_masked_timesteps=0)

                img_name = f"MD_{MODEL_ID}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')



                # We'll just visualize the prediction in the first batch
                for batch in test_dataloader:
                    Y, Y_hat, mask = trainer.model.forward(batch)
                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()


                # Use first batch for visualization (safe indexing - always use batch 0)
                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)

                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()



                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"MD_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_lstmvae(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(
            ParameterGrid(LSTMVAE_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(LSTMVAE_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model_hyper_params['n_features'] = self.train_data.n_features
                training_args = TrainingArguments(**train_hyper_params)
                model = LSTMVAE(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"LSTMVAE_{MODEL_ID+1}"):
                        print(f'Model LSTMVAE_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1

                # eval the model
                trainer.model.eval()
                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0
                )

                img_name = f"LSTMVAE_{MODEL_ID}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')
                # We'll just visualize the prediction in the first batch
                for batch in test_dataloader:
                    Y, Y_mu, mask, Y_sigma, Z_mu, Z_sigma, Z = trainer.model.forward(batch)
                    Y, Y_mu, mask, Y_sigma, Z_mu, Z_sigma, Z = Y.detach().cpu().numpy(), Y_mu.detach().cpu().numpy(), mask.detach().cpu().numpy(), Y_sigma.detach().cpu().numpy(), Z_mu.detach().cpu().numpy(), Z_sigma.detach().cpu().numpy(), Z.detach().cpu().numpy()
                    break

                batch_num = 0  # Use first batch (safe for any batch size)
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_mu[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.fill_between(x=np.arange(len(Y_mu[batch_num, feature_num, :])), y1=Y_mu[batch_num, feature_num, :]+Y_sigma[batch_num, feature_num, :], y2=Y_mu[batch_num, feature_num, :]-Y_sigma[batch_num, feature_num, :], color='r', alpha=0.3)
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"LSTMVAE_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_rnn(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(RNN_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(RNN_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                training_args = TrainingArguments(**train_hyper_params)
                model = RNN(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"RNN_{MODEL_ID+1}"):
                        print(f'Model RNN_{MODEL_ID+1} already trained!')
                        continue

                trainer = Trainer(model=model,
                                  args=training_args,
                                  train_dataset=self.train_data,
                                  eval_dataset=None,
                                  verbose=self.verbose)
                trainer.train()
                MODEL_ID = MODEL_ID + 1

                # eval the model
                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0
                )

                img_name = f"RNN_{MODEL_ID}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                # We'll just visualize the prediction in the first batch
                for batch in test_dataloader:
                    batch_size, n_features, window_size = batch['Y'].shape
                    Y, Y_hat, mask = trainer.model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()
                    Y = Y.reshape(n_features, -1)[:, :window_size]  # to [n_features, n_time]
                    Y_hat = Y_hat.reshape(n_features, -1)[:, :window_size]  # to [n_features, n_time]
                    mask = mask.reshape(n_features, -1)[:, :window_size]  # to [n_features, n_time]

                    break

                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[feature_num, :512], c='darkblue', label='Y')
                axes.plot(Y_hat[feature_num, :512], c='red', label='Y_hat')
                axes.legend(fontsize=16)

                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()



                # Save the model
                self.logging_obj.save(obj=trainer.model,
                                      obj_name=f"RNN_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_rm(self):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(RM_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(RM_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = RunningMean(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"RM_{MODEL_ID+1}"):
                        print(f'Model RM_{MODEL_ID+1} already trained!')
                        continue

                MODEL_ID = MODEL_ID + 1

                # eval the model
                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0
                )

                img_name = f"RM_{MODEL_ID}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')


                # We'll just visualize the prediction in the first batch
                for batch in test_dataloader:
                    Y, Y_hat, mask = model.forward(batch)
                    break

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"RM_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_nn(self, batch_size=32):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(NN_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(NN_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = NearestNeighbors(**model_hyper_params)
                model_hyper_params_str = json.dumps(model_hyper_params)
                print(f"Model NN_{MODEL_ID+1} Params")
                with open('output1.txt', 'a') as file:
                    # Write data to the file
                    file.write(model_hyper_params_str)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"NN_{MODEL_ID+1}"):
                        print(f'Model NN_{MODEL_ID+1} already trained!')

                        continue

                train_dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(train_dataloader)

                MODEL_ID = MODEL_ID + 1


                # eval the model

                test_dataloader = Loader(dataset=self.test_data,
                                    batch_size=8,
                                    window_size=64,
                                    window_step=64,
                                    shuffle=False,
                                    padding_type='right',
                                    sample_with_replace=False,
                                    verbose=False,
                                    mask_position='None',
                                    n_masked_timesteps=0)

                img_name = f"NN_{MODEL_ID}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                # We'll just visualize the prediction in the first batch
                for batch in test_dataloader:
                    Y, Y_hat, mask = model.forward(batch)
                    break

                batch_num = 0  # Use first batch (safe for any batch size)
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()





                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"NN_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                          train_hyper_params,
                                          'model_hyperparameters':
                                          model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_sos(self, batch_size=32):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(SOS_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(SOS_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadSOS(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"SOS_{MODEL_ID + 1}"):
                        print(f'Model SOS_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"SOS_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self._diagnostic_batch_size(
                        'SOS', model_hyper_params['window_size']),
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"SOS_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_alad(self, batch_size=32):
        MODEL_ID = 0
        model_hyper_param_configurations = list(ParameterGrid(ALAD_PARAM_GRID))
        train_hyper_param_configurations = list(
            ParameterGrid(ALAD_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                model = TsadALAD(**model_hyper_params)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"ALAD_{MODEL_ID + 1}"):
                        print(f'Model ALAD_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"ALAD_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self.batch_size,
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"ALAD_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

    def train_pyod(self, model_name: str, batch_size=32):
        # A family may bring its own grid: the PyOD 3 time-series detectors need
        # a different subsequence length and a lower epoch count than the shared
        # contamination sweep, and AutoEncoder varies its architecture.
        # Everything else keeps PYOD_PARAM_GRID.
        grid = PYOD_MODEL_GRIDS.get(model_name.upper(), PYOD_PARAM_GRID)
        self._train_wrapped(PyodModel, model_name, grid, batch_size)

    def train_tsbad(self, model_name: str, batch_size=32):
        """The TSB-AD families, over the vendored code in `Algorithms/tsb_ad`.

        Identical to `train_pyod` but for the wrapper class and the grid table:
        `TSBADModel` takes the same constructor arguments as `PyodModel` exactly
        so one loop can train both. Every TSB-AD family must have its own grid —
        no two of these detectors take the same parameters, so there is no
        shared default to fall back to.
        """
        grid = TSBAD_MODEL_GRIDS.get(model_name.upper())
        if grid is None:
            raise ValueError(
                f"No hyperparameter grid for TSB-AD family {model_name}. "
                f"Known: {', '.join(sorted(TSBAD_MODEL_GRIDS))}")
        self._train_wrapped(TSBADModel, model_name, grid, batch_size)

    def _train_wrapped(self, model_cls, model_name: str, grid: dict, batch_size=32):
        """Train every instance of a family whose detector lives behind a wrapper.

        Shared by `train_pyod` and `train_tsbad`, which differ only in which
        wrapper class they build and where they look the grid up. Both wrappers
        take `(model_name, **framework_params, detector_kwargs=...)`.
        """
        MODEL_ID = 0
        upper_model_name = model_name.upper()
        model_hyper_param_configurations = list(ParameterGrid(grid))
        train_hyper_param_configurations = list(
            ParameterGrid(PYOD_TRAIN_PARAM_GRID))
        for train_hyper_params in tqdm(train_hyper_param_configurations):
            for model_hyper_params in tqdm(model_hyper_param_configurations):
                # `detector__x` is the PyOD estimator's own parameter x; the rest
                # are the framework's. Both spell some of them the same way.
                framework = {k: v for k, v in model_hyper_params.items()
                             if not k.startswith('detector__')}
                detector = {k[len('detector__'):]: v for k, v in model_hyper_params.items()
                            if k.startswith('detector__')}
                model = model_cls(model_name, **framework, detector_kwargs=detector)

                if not self.overwrite:
                    if self.logging_obj.check_file_exists(
                            obj_class=self.logging_hierarchy,
                            obj_name=f"{upper_model_name}_{MODEL_ID + 1}"):
                        print(f'Model {upper_model_name}_{MODEL_ID + 1} already trained!')
                        continue

                dataloader = Loader(
                    dataset=self.train_data,
                    batch_size=batch_size,
                    window_size=model_hyper_params['window_size'],
                    window_step=model_hyper_params['window_step'],
                    shuffle=False,
                    padding_type='right',
                    sample_with_replace=False,
                    verbose=False,
                    mask_position='None',
                    n_masked_timesteps=0)
                model.fit(dataloader)

                img_name = f"{upper_model_name}_{MODEL_ID + 1}.png"
                img_path = os.path.join(self.img_dir, img_name)
                logger.info(f'img_path is {img_path} ')

                test_dataloader = Loader(
                    dataset=self.test_data,
                    batch_size=self._diagnostic_batch_size(
                        model_name, model_hyper_params['window_size']),
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
                    Y, Y_hat, mask = model.forward(batch)

                    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(), mask.detach().cpu().numpy()

                batch_num = 0
                feature_num = 0
                fig, axes = plt.subplots(1, 1, sharey=True, figsize=(15, 4))
                axes.plot(Y[batch_num, feature_num, :].flatten(), c='darkblue', label='Y')
                axes.plot(Y_hat[batch_num, feature_num, :].flatten(), c='red', label='Y_hat')
                axes.legend(fontsize=16)
                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()

                MODEL_ID = MODEL_ID + 1
                # Save the model
                self.logging_obj.save(obj=model,
                                      obj_name=f"{upper_model_name}_{MODEL_ID}",
                                      obj_meta={
                                          'train_hyperparameters':
                                              train_hyper_params,
                                          'model_hyperparameters':
                                              model_hyper_params
                                      },
                                      obj_class=self.logging_hierarchy)

