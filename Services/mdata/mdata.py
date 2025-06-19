from typing import Optional
import os
from loguru import logger
from utils.utils import get_args_from_cmdline

class Mdata():
    def __init__(self,_dataset_type:Optional[str]=None,_data_entity:Optional[str]=None):
        args = get_args_from_cmdline()
        self._dataset_type = _dataset_type
        self._data_entity = _data_entity
        self.result_dir = args['results_path']
        self.trained_model_dir = args['trained_model_path']


    def delete_data_cache(self):

        data_models_dir = os.path.join(self.trained_model_dir,f'{self._dataset_type}/{self._data_entity}')
        files_list = []
        for root, dirs, files in os.walk(data_models_dir):
            files_list = files
            break

        for file in files_list:
            data_file_path = os.path.join(data_models_dir,file)
            # logger.info(f'data_file_path is {data_file_path}')
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            logger.info(f'--------------remove {data_file_path}-----------')

        data_result_dir = os.path.join(self.result_dir, f'{self._dataset_type}')


        data_obj_result_path = os.path.join(data_result_dir,f'ranking_obj_{self._data_entity}.data')
        if os.path.exists(data_obj_result_path):
            os.remove(data_obj_result_path)
        logger.info(f'--------------remove {data_obj_result_path}-----------')

        data_img_result_dir = os.path.join(data_result_dir, f'img_{self._data_entity}')

        files_list = []
        for root, dirs, files in os.walk(data_img_result_dir):
            files_list = files
            break

        for file in files_list:
            data_file_path = os.path.join(data_img_result_dir, file)
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            logger.info(f'--------------remove {data_file_path}-----------')






