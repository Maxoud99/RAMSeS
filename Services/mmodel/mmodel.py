

from loguru import logger
import traceback
from typing import Optional,List

from Model_Training.train import TrainModels

from Utils.utils import get_args_from_cmdline
from Utils.set_all_seeds import set_all_seeds
from dao.mdata.mdata import update_data_status_by_name

class Mmodel(object):

    def __init__(self,dataset:Optional[str]=None,entities:Optional[str]=None,algorithm_list:List[str]=[]):
        self.dataset = dataset
        self.entities = entities
        self.algorithm_list = algorithm_list



    def train(self):
        args = get_args_from_cmdline()

        logger.info('Set all seeds!')
        set_all_seeds(args['random_seed'])

        logger.info(42 * "=")
        logger.info(f"Training models on entity: {self.entities}")
        logger.info(42 * "=")
        model_trainer = TrainModels(dataset=self.dataset,
                                    entity=self.entities,
                                    algorithm_list=self.algorithm_list,
                                    downsampling=args['downsampling'],
                                    min_length=args['min_length'],
                                    root_dir=args['dataset_path'],
                                    training_size=args['training_size'],
                                    overwrite=args['overwrite'],
                                    verbose=args['verbose'],
                                    save_dir=args['trained_model_path'])
        try:
            model_trainer.train_models(model_architectures=args['model_architectures'])
            update_data_status_by_name(_data_name=self.entities,_status=2)
        except:
            logger.info(f'Traceback for Entity: {self.entities} Dataset: {self.dataset}')
            print(traceback.format_exc())
        logger.info(42 * "=")

