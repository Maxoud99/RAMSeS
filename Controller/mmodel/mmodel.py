import pkgutil

from flask import Blueprint, request, render_template, render_template_string, jsonify, redirect, url_for
from loguru import logger
import os
import templates
import pyod.models as pyod_models
from dao.mmodel.mmodel import delete_algorithm_name
from dao.mmodel.mmodel import insert_model_name
from dao.mdata.mdata import update_data_status_by_name,update_data_algorithm_by_name,select_data_entity_by_status
from services.mmodel.mmodel import Mmodel
from dao.mdata.mdata import query_data_type,select_algorithms_by_data_entity
from dao.mmodel.mmodel import query_algorithm_name
from utils.utils import get_args_from_cmdline,img2binary
mmodel_bp = Blueprint("mmodel",__name__,url_prefix="/mmodel")

from concurrent.futures import ProcessPoolExecutor


# train the model
@mmodel_bp.route('/train',methods = ['GET','POST'])
def train():
    logger.info(f'request.method is {request.method}')
    if request.method == 'POST':
        dataset_selection = request.form.getlist('dataset_selection')[0]
        dataset_entity_list = request.form.getlist('dataset_entity_selection')
        algorithm_list = request.form.getlist('algorithm_option')
        logger.info(f'dataset_selection is {dataset_selection}\ndataset_entity_list is {dataset_entity_list}\nalgorithm_list is {algorithm_list}')

        #TODO data set entity can be list
        for dataset_entity in dataset_entity_list:
            update_data_status_by_name(_data_name=dataset_entity,_status=1)
            algorithms = '_'.join(algorithm_list)
            logger.info(f'Algorithms is {algorithms}')
            update_data_algorithm_by_name(_data_name=dataset_entity,_algorithm=algorithms)

        data_entity_info_list = select_data_entity_by_status(_status=1)
        data_infos = []
        for data_entity_info in data_entity_info_list:
            data_info = {'dataset_entity': data_entity_info[0], 'Algorithms': data_entity_info[1],
                         'dataset_type': data_entity_info[2], 'inject_abn_types': data_entity_info[3],
                         'best_model': data_entity_info[4]}
            data_infos.append(data_info)
        # algorithm_list = query_algorithm_name()

        mmodel_instance = Mmodel(dataset=dataset_selection,entities=dataset_entity_list[0],algorithm_list=algorithm_list)

        # background running
        executor = ProcessPoolExecutor(1)
        executor.submit(mmodel_instance.train,)
        # mmodel_instance.train()

        return render_template(template_name_or_list='index.html',data_infos = data_infos,algorithm_list = algorithm_list,table_flag = True)

    # flash the page,return data set type and algorithm list
    if request.method == 'GET':
        dataset_types = query_data_type()

        algorithm_list = query_algorithm_name()
        return render_template(template_name_or_list='index.html', dataset_types=dataset_types,
                               algorithm_list=algorithm_list)

# check train result
@mmodel_bp.route('/check')
def check_train_result():

    # request from href
    if request.method == 'GET':
        dataset_type = request.args.get('dataset_type')
        dataset_entity = request.args.get('dataset_entity')

        logger.info(f'mmodel check dataset_type is {dataset_type},dataset_entity is {dataset_entity}')

        args = get_args_from_cmdline()
        train_model_dir = args['trained_model_path']
        s_train_model_dir = os.path.join(train_model_dir,f'{dataset_type}/{dataset_entity}')
        logger.info(f's_train_model_dir is {s_train_model_dir}')

        files_list = []

        for root, dirs, files in os.walk(s_train_model_dir):

            files_list = files
            break

        logger.info(f'files_list is {files_list}')

        img_html = "<legend>Model Detail</legend>"

        # if ori data already generated
        if 'data.png' in files_list:
            data_img_path = os.path.join(s_train_model_dir,'data.png')
            save_file_base64 = img2binary(img_path=data_img_path)
            data_img_html = """
            <fieldset>
            <legend>{}</legend>
                    <table cellspacing="0" cellpadding="5" width="96%" style="margin-left:2.2%;width: 96% !important; min-width: 96%; max-width: 96%;">
                        <td><br><span style="color:blue;font-size:6px" align="left">&emsp;&emsp;</span><br><img src = "data:image/png;base64,{}"></td>
                    </table>
            </fieldset>
                    """.format('original data',save_file_base64)
            img_html += data_img_html

        # filter predict error img
        algorithm_id_list = []
        img_files_list = [file for file in files_list if (file.endswith('.png')) & (not file.startswith('data'))]
        logger.info(f'img_files_list is {img_files_list}')


        # first into page,if not yet select data and algorithm,so select cache img
        algorithms = select_algorithms_by_data_entity(_data_name=dataset_entity)
        if  algorithms != None:
            for file in img_files_list:
                for algorithm in algorithms.split('_'):
                    if algorithm in file :
                        algorithm_id_list.append(file)

        # if Algorithms is none,indicate first train
        else:
            for file in img_files_list:
              algorithm_id_list.append(file)

        logger.info(f'algorithm_id_list is {algorithm_id_list}')


        for algorithm_id in algorithm_id_list:
            model_img_path = os.path.join(s_train_model_dir, algorithm_id)
            logger.info(f'model_img_path is {model_img_path}')
            save_file_base64 = img2binary(model_img_path)
            model_img_html ="""
            <fieldset>
            <legend>{}</legend>
                               <table cellspacing="0" cellpadding="5" width="96%" style="margin-left:2.2%;width: 96% !important; min-width: 96%; max-width: 96%;">
                                   <td><br><span style="color:blue;font-size:6px" align="left">&emsp;&emsp;</span><br><img src = "data:image/png;base64,{}"></td>
                               </table>
            </fieldset>
                               """.format(algorithm_id,save_file_base64)
            img_html += model_img_html


        return  img_html

@mmodel_bp.route('/add-model', methods=['POST'])
def add_model():
    data = request.get_json()  # Get JSON data from the request
    new_model_name = str(data['newModelName']).upper()
    # Code to insert the new model name into your database
    # For example: insert_into_database(new_model_name)
    module_names = []
    for _, module_name, _ in pkgutil.walk_packages(pyod_models.__path__, prefix=pyod_models.__name__ + '.'):
        module_names.append(module_name)

    pyod_model_name = f'pyod.models.{new_model_name.lower()}'
    if pyod_model_name not in module_names:
        # Return a response that includes a flag for showing the alert
        return jsonify({'exists': False, 'message': 'I can only add models from pyod library for now'})
    else:
        # If it does, insert the new model and indicate success
        is_success = insert_model_name(new_model_name)  # Replace with your function to insert the model
        if is_success:
            message = 'Model added successfully from pyod library.'
        else:
            message = 'Model already exists'
        return jsonify({'exists': True, 'message': message})


@mmodel_bp.route('/delete-algorithm', methods=['POST'])
def delete_algorithm():
    algorithm_name = request.form['algorithm_name']
    if delete_algorithm_name(algorithm_name):
        return jsonify({'message': 'Algorithm deleted successfully'}), 200
    else:
        return jsonify({'error': 'Algorithm not found'}), 404


