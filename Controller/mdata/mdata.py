

from flask import  Blueprint,render_template,request,template_rendered,redirect
from loguru import logger
mdata_bp = Blueprint("mdata",__name__,url_prefix="/mdata")


from dao.mdata.mdata import query_data_type,query_data_by_type,delete_data_by_data_entity
from services.mdata.mdata import Mdata

# get data set type
@mdata_bp.route('/getDataSetType',methods = ['GET','POST'])
def get_dataset_type():
    if request.method == 'GET':

        data_types = query_data_type()
        logger.info(f'data_types is {data_types}')


        return render_template(template_name_or_list='mdata/mdata.html',data_types = data_types)


# get data set entity html
@mdata_bp.route('/getDataSetEntityHtml',methods = ['GET','POST'])
def get_dataset_entity_html():
    if request.method == 'POST':

        dataset_type = request.form.get('dataset_type')
        print(f'select data_type is {dataset_type}')
        data_entity_list = query_data_by_type(_data_type=dataset_type)

        options = ''.join([f'<option value="{data_entity}">{data_entity}</option>' for data_entity in data_entity_list])
        print(f'options is {options}')
        return options



# delete data info
@mdata_bp.route('/deleteDataInfo',methods = ['GET','POST'])
def delete_data_info():
    if request.method == 'GET':

        dataset_type = request.args.get('dataset_type')
        data_entity = request.args.get('dataset_entity')
        mdata_instance = Mdata(_dataset_type=dataset_type,_data_entity=data_entity)
        logger.info(f'delete_data_info data_entity is {data_entity}')
        delete_data_by_data_entity(_data_name=data_entity)
        mdata_instance.delete_data_cache()

        return redirect(location='/')