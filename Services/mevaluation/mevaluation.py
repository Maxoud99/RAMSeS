
import numpy as np
from typing import Optional,List
import matplotlib.pyplot  as plt
import os
from loguru import logger


from Datasets.load import load_data
from Model_Selection.model_selection import RankModels
# from Model_Selection.inject_anomalies import InjectAnomalies
from Utils.utils import get_args_from_cmdline
from Utils.logger import Logger
from Utils.utils import visualize_data
from dao.mdata.mdata import update_data_status_by_name,select_algorithms_by_data_entity,select_inject_abn_types_by_data_entity
# from Model_Selection.model_selection import RankModels

class Mevaluation(object):


    def __init__(self):
        args = get_args_from_cmdline()
        self.data_dir = args['dataset_path']
        self.result_dir = args['results_path']
        self.trained_model_path = args['trained_model_path']
        self.overwrite = args['overwrite']
        self.verbose = args['verbose']




    def inject_abnormal(self,_dataset_type:Optional[str]=None,_dataset_entity:Optional[str]=None,abn_type_list:List[str]=None):


        img_dir = os.path.join(self.result_dir,f'{_dataset_type}/img_{_dataset_entity}')
        os.path.exists(img_dir) or os.makedirs(img_dir)
        logger.info(f'img_dir is {img_dir}')
        train_data = load_data(dataset=_dataset_type, group='train', entities=[_dataset_entity], downsampling=10, min_length=256,
                               root_dir=self.data_dir, normalize=True, verbose=False)
        test_data = load_data(dataset=_dataset_type, group='test', entities=[_dataset_entity], downsampling=10, min_length=256,
                              root_dir=self.data_dir, normalize=True, verbose=False)
        data_img_name = f'{_dataset_entity}_data.png'
        save_path = os.path.join(img_dir,data_img_name)

        os.path.exists(save_path) or visualize_data(train_data, test_data,savefig=True,save_path=save_path)
        ## Now let's inject anomalies to the data and visualize it

        T = train_data.entities[0].Y

        anomalyObj = InjectAnomalies(random_state=np.random.randint(10000),
                                     verbose=True,
                                     max_window_size=128,
                                     min_window_size=8)
        data_std = max(np.std(T), 0.01)

        for abn_type in abn_type_list:
            abn_img_name = f'{_dataset_entity}_{abn_type}.png'
            logger.info(f'abn_img_name is {abn_img_name}')
            img_path = os.path.join(img_dir, abn_img_name)


            if not os.path.exists(img_path):


                T_with_anomaly, anomaly_sizes, anomaly_labels = anomalyObj.inject_anomalies(T=T,
                                                                scale=2*data_std,
                                                                anomaly_type=abn_type)

                anomaly_start = np.argmax(anomaly_labels)
                anomaly_end = T_with_anomaly.shape[1] - np.argmax(anomaly_labels[::-1])

                # anomaly_sizes = anomaly_sizes/data_std

                fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
                axes[0].plot(T_with_anomaly.flatten(), color='darkblue')
                axes[0].plot(np.arange(anomaly_start, anomaly_end), T_with_anomaly.flatten()[anomaly_start:anomaly_end],
                             color='red')
                axes[0].plot(np.arange(anomaly_start, anomaly_end), T.flatten()[anomaly_start:anomaly_end], color='darkblue',
                             linestyle='--')
                axes[0].set_title('Train data with Injected Anomalies', fontsize=16)
                axes[1].plot(anomaly_sizes.flatten(), color='pink')
                axes[1].plot(anomaly_labels.flatten(), color='red')
                axes[1].set_title('Anomaly Scores', fontsize=16)


                plt.savefig(img_path, format='png')
                plt.clf()
                plt.close()
                logger.info(f'save abn img {abn_img_name}')

            else:
                logger.info(f'{abn_img_name} already exist in {img_dir}')

        update_data_status_by_name(_data_name=_dataset_entity,_status=3)



    def evaluate_model(self,_dataset_type:Optional[str]=None,_dataset_entity:Optional[str]=None):

        # Algorithms = select_algorithms_by_data_entity(_data_name=_dataset_entity)
        inject_abn_types = select_inject_abn_types_by_data_entity(_data_name=_dataset_entity)
        inject_abn_list = inject_abn_types.split('_')
        algorithms = select_algorithms_by_data_entity(_data_name=_dataset_entity)
        model_name_list = algorithms.split('_')
        logger.info(f'evaluate_model method inject_abn_list is {inject_abn_list},model_name_list is {model_name_list}')
        rank_model_params = {
            'dataset': _dataset_type,  # anomaly_archive
            'entity': _dataset_entity,  # '001_UCR_Anomaly_DISTORTED1sddb40',
            'inject_abn_list':inject_abn_list,
            'model_name_list':model_name_list,
            'trained_model_path': self.trained_model_path,
            'downsampling': 10,
            'min_length': 256,
            'root_dir': self.data_dir,
            'normalize': True,
            'verbose': False
        }

        rankingObj = RankModels(**rank_model_params)

        _ = rankingObj.evaluate_models(n_repeats=1, n_neighbors=[4], split='test', synthetic_ranking_criterion='f1', n_splits=100)

        logging_obj = Logger(save_dir=self.result_dir,
                             overwrite=self.overwrite,
                             verbose=self.verbose)

        logging_obj.save(obj=rankingObj,
                         obj_name=f'ranking_obj_{_dataset_entity}',
                         obj_meta=None,
                         obj_class=[_dataset_type],
                         type='data')
