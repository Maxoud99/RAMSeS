from flask import  Blueprint,request,render_template

from loguru import logger

import os

import joblib

import numpy as np

import pandas as pd

from Services.mevaluation.mevaluation import Mevaluation

from Utils.utils import get_args_from_cmdline,img2binary

from dao.mdata.mdata import update_data_inject_abn_types_by_name,select_inject_abn_types_by_data_entity,select_data_status_by_entity,update_data_best_model_by_name

args = get_args_from_cmdline()

from Model_Selection.rank_aggregation import trimmed_borda, trimmed_partial_borda, borda, kemeny, partial_borda, influence, averagedistance, average_rank_aggregator, enhanced_markov_chain_rank_aggregator, copeland_rank_aggregator, spearmans_footrule_aggregator

from Utils.model_selection_utils import rank_models


mevaluation_bp = Blueprint("mevaluation",__name__,url_prefix="/mevaluation")



from concurrent.futures import ProcessPoolExecutor

# inject abnormal
@mevaluation_bp.route('/injectAbn',methods = ['GET','POST'])
def inject_abn():
    mevaluation_instance = Mevaluation()

    abn_type_list = args['abn_type_list']
    if request.method == 'GET':
        dataset_type = request.args.get('dataset_type')
        dataset_entity = request.args.get('dataset_entity')
        algorithms = request.args.get('Algorithms')
        flash_flag = request.args.get('flash_flag','False')
        logger.info(
            f'dataset_type is {dataset_type},dataset_entity is {dataset_entity},Algorithms is {algorithms},flash_flag is {flash_flag}')

        logger.info(f'GET abn_type_list is {abn_type_list}')

        selected_inject_abn_types = select_inject_abn_types_by_data_entity(_data_name=dataset_entity)

        # request from flash or the data entity have not select inject abn type,return pure page
        if (flash_flag== 'True') | (selected_inject_abn_types == None):
            logger.info(f'flash_flag false')
            return render_template('mevaluation/inject_abn.html', abn_type_list=abn_type_list,
                                   dataset_type=dataset_type, dataset_entity=dataset_entity)
        # return select inject abn type page
        else:
            logger.info(f'flash_flag false')
            abn_type_list = selected_inject_abn_types.split('_')
            return render_template('mevaluation/inject_abn.html',abn_types = selected_inject_abn_types ,abn_type_list=abn_type_list,dataset_type=dataset_type, dataset_entity=dataset_entity,search_abn_img = True)


    # when submit the form,flash the page
    if request.method == 'POST':
        dataset_type = request.form.get('dataset_type')
        dataset_entity = request.form.get('dataset_entity')
        abn_type_list = request.form.getlist('abn_option')
        abn_types = '_'.join(abn_type_list)
        try:
            update_data_inject_abn_types_by_name(_data_name=dataset_entity,_inject_abn_types=abn_types)
            logger.info(f'update data inject abn type success')
        except:
            logger.error(f'update data inject abn type error')


        logger.info(
            f'evaluate_model dataset_type is {dataset_type},dataset_entity is {dataset_entity},abn_type_list is {abn_type_list}')

        # inject abnormal running in background
        executor = ProcessPoolExecutor(1)
        executor.submit(mevaluation_instance.inject_abnormal,dataset_type,dataset_entity,abn_type_list)

        #mevaluation_instance.inject_abnormal(_dataset_type=dataset_type,_dataset_entity=dataset_entity,abn_type_list=abn_type_list)

        return render_template('mevaluation/inject_abn.html',abn_types = abn_types,abn_type_list = abn_type_list,dataset_type = dataset_type,dataset_entity = dataset_entity,search_abn_img = True)

# search inject abnormal result img
@mevaluation_bp.route('/searchImg',methods = ['GET','POST'])
def search_img():

    # request from ajax
    if request.method == 'POST':
        dataset_type = request.form.get('dataset_type')
        dataset_entity = request.form.get('dataset_entity')
        abn_types = request.form.get('abn_types')

        logger.info(f'search_img dataset_type is {dataset_type},dataset_entity is {dataset_entity},abn_types is {abn_types}')
        result_dir = args['results_path']

        dataset_entity_result_dir = os.path.join(result_dir,f'{dataset_type}/img_{dataset_entity}')

        files_list = []

        for root, dirs, files in os.walk(dataset_entity_result_dir):
            files_list = files
            break

        img_html = "<legend>Inject Abnormal Detail</legend>"
        data_img_name = f'{dataset_entity}_data.png'
        logger.info(f'data_img_name is {data_img_name}')

        # if ori data img already generate
        if data_img_name in files_list:

            data_img_path = os.path.join(dataset_entity_result_dir, data_img_name)
            logger.info(f'data_img_path is {data_img_path}')
            save_file_base64 = img2binary(img_path=data_img_path)
            data_img_html = """
                    <fieldset>
                    <legend>{}</legend>
                            <table cellspacing="0" cellpadding="5" width="96%" style="margin-left:2.2%;width: 96% !important; min-width: 96%; max-width: 96%;">
                                <td><br><span style="color:blue;font-size:6px" align="left">&emsp;&emsp;</span><br><img src = "data:image/png;base64,{}"></td>
                            </table>
                    </fieldset>
                            """.format('train and test data', save_file_base64)
            img_html += data_img_html

        # if abn type img already generate
        for abn_type in abn_types.split('_'):
            abn_type_img_name = f'{dataset_entity}_{abn_type}.png'
            logger.info(f'abn_type_img_name is {abn_type_img_name}')
            if abn_type_img_name in files_list:
                abn_type_img_path = os.path.join(dataset_entity_result_dir, abn_type_img_name)
                logger.info(f'abn_type_img_path is {abn_type_img_path}')
                save_file_base64 = img2binary(img_path=abn_type_img_path)
                abn_type_img_html = """
                                    <fieldset>
                                    <legend>{}</legend>
                                            <table cellspacing="0" cellpadding="5" width="96%" style="margin-left:2.2%;width: 96% !important; min-width: 96%; max-width: 96%;">
                                                <td><br><span style="color:blue;font-size:6px" align="left">&emsp;&emsp;</span><br><img src = "data:image/png;base64,{}"></td>
                                            </table>
                                    </fieldset>
                                            """.format(abn_type, save_file_base64)
                img_html += abn_type_img_html



        return img_html


    if request.method == 'GET':
        pass









# evaluate all model
@mevaluation_bp.route('/evaluateModel',methods = ['GET','POST'])
def evaluate_model():
    mevaluation_instance = Mevaluation()
    executor = ProcessPoolExecutor(1)
    if request.method == 'GET':

        dataset_type = request.args.get('dataset_type')
        dataset_entity = request.args.get('dataset_entity')
        logger.info(
            f'evaluateModel dataset_type is {dataset_type},dataset_entity is {dataset_entity}')

        data_status = select_data_status_by_entity(_data_name=dataset_entity)
        inject_abn_types = select_inject_abn_types_by_data_entity(_data_name=dataset_entity)
        if inject_abn_types==None:
            return '''<h1>Did not complete abnormal inject</h1>'''

        inject_type_list = inject_abn_types.split('_')
        logger.info(f'evaluateModel inject_type_list is {inject_type_list}')
        # if status is 1,indicate model is training
        if data_status == 1:
            return '''<h1>Did not complete train</h1>'''
        # if status is 2,and have not complate inject abnormal
        elif (data_status == 2) & (len(inject_type_list)<1):
            return '''<h1>Did not complete abnormal inject</h1>'''
        else:
            result_dir = args['results_path']
            result_obj_path = os.path.join(result_dir,f'{dataset_type}/ranking_obj_{dataset_entity}.data')
            logger.info(f'result_obj_path is {result_obj_path}')


            if not os.path.exists(result_obj_path):

                executor.submit(mevaluation_instance.evaluate_model, dataset_type, dataset_entity)
                return render_template(template_name_or_list='mevaluation/evaluate_model.html',dataset_type = dataset_type,dataset_entity = dataset_entity)
            else:
                return render_template(template_name_or_list='mevaluation/evaluate_model.html',
                                       dataset_type=dataset_type, dataset_entity=dataset_entity)




# revaluate all model
@mevaluation_bp.route('/revaluateModel',methods = ['GET','POST'])
def revaluate_model():
    mevaluation_instance = Mevaluation()
    executor = ProcessPoolExecutor(1)
    if request.method == 'POST':
        dataset_type = request.form.get('dataset_type')
        dataset_entity = request.form.get('dataset_entity')
        logger.info(
            f'revaluateModel dataset_type is {dataset_type},dataset_entity is {dataset_entity}')

        result_dir = args['results_path']

        result_obj_path = os.path.join(result_dir, f'{dataset_type}/ranking_obj_{dataset_entity}.data')

        logger.info(f'result_obj_path is {result_obj_path}')
        if os.path.exists(result_obj_path):
            os.remove(result_obj_path)
            logger.info(f'remove {result_obj_path}')


        executor.submit(mevaluation_instance.evaluate_model, dataset_type, dataset_entity)

        return render_template(template_name_or_list='mevaluation/evaluate_model.html',

                               dataset_type=dataset_type, dataset_entity=dataset_entity)


# search evaluate img
@mevaluation_bp.route('/searchEvaluateImg',methods = ['GET','POST'])
def search_evaluate_result():
    # request from ajax
    if request.method == 'POST':

        dataset_type = request.form.get('dataset_type')
        dataset_entity = request.form.get('dataset_entity')
        logger.info(
            f'searchEvaluateImg dataset_type is {dataset_type},dataset_entity is {dataset_entity}')
        result_dir = args['results_path']

        result_obj_path = os.path.join(result_dir,f'{dataset_type}/ranking_obj_{dataset_entity}.data')

        logger.info(f'result_obj_path is {result_obj_path}')


        result_html = ''



        # if have not generated evaluate result path
        if not os.path.exists(result_obj_path):
            result_html += '''<p>Model evaluation is not yet complete</p>'''
            return result_html

        else:

            rankingObj = joblib.load(result_obj_path)

            # Models Performance Matrix
            models_performance_matrix = rankingObj.models_performance_matrix
            models_performance_matrix_html = models_performance_matrix.to_html()

            ranks_by_metrics, rank_prauc, rank_f1, rank_vus = rank_models(models_performance_matrix)

            n_metrics, n_models = ranks_by_metrics.shape
            logger.info(f'n_metrics is {n_metrics},n_models is {n_models}')

            result_html += ''' <fieldset>
                <p>Number of metrics: {} | Number of models: {}</p>'''.format(str(n_metrics),str(n_models))
            result_html += '''</fieldset>'''





            result_html += ''' <fieldset>
                <legend>Models Performance Matrix</legend>'''
            result_html += models_performance_matrix_html
            result_html += '''</fieldset>'''




            # Models Rank Matrix
            model_names = models_performance_matrix.index.values
            logger.info(f'model_names is {model_names}')



            # Models Rank Matrix
            ranks = np.concatenate([ranks_by_metrics[:6, :], ranks_by_metrics[6::2, :]], axis=0).astype(int)
            logger.info(f'ranks is {ranks}')
            # df_rank = pd.DataFrame(ranks, columns=model_names, index=model_names)

            # model_rank_html = df_rank.to_html()
            #
            # result_html += ''' <fieldset>
            #     <legend>Models Rank Matrix</legend>'''
            # result_html += model_rank_html
            # result_html += '''</fieldset>'''


            # Rank F1
            f1 = models_performance_matrix.iloc[:, 1].to_numpy()
            model_rank_f1 = f1[rank_f1.astype(int)]
            logger.info(f'model_rank_f1 is {model_rank_f1}')
            rank_column_name_list = ['rank']
            df_model_rank_f1 = pd.DataFrame(model_rank_f1,index=model_names,columns=rank_column_name_list)
            df_model_rank_f1_html = df_model_rank_f1.to_html()

            result_html += ''' <fieldset>
                        <legend>Models Rank Matrix (f1) </legend>'''
            result_html += df_model_rank_f1_html
            result_html += '''</fieldset>'''


            # test ###################################################
            logger.info(f'ranks_by_metrics is {ranks_by_metrics}')
            avg_objective, avg_aggregated_rank = average_rank_aggregator(ranks_by_metrics)
            logger.info(f'avg_rank is {avg_aggregated_rank}')
            avg_rank_df = pd.DataFrame(avg_aggregated_rank,index=model_names,columns=rank_column_name_list)
            average = avg_rank_df.to_html()
            result_html += ''' <fieldset>
                        <legend>average rank values </legend>'''
            result_html += average
            result_html += '''</fieldset>'''
            logger.info(f'f1 shape is {f1}')

            # end ###################################################
            #                            *                          #
            #                           * *                         #
            #                          * * *                        #
            #                           * *                         #
            #                            *                          #
            # start again ###########################################

            rank_arrays = [ranks_by_metrics]
            for rank_array in [rank_prauc, rank_f1, rank_vus]:
                if rank_array.ndim == 1:
                    rank_array = rank_array.reshape(1, -1)
                rank_arrays.append(rank_array)

            f1_scores = models_performance_matrix.iloc[:, 1].to_numpy()
            logger.info(f'f1_ranks before markov is {f1_scores}')
            logger.info(f'rank_arrays for markov is {rank_arrays}')
            # Convert F1 scores to ranks - the highest score gets rank 1 and so on
            f1_ranks = f1_scores.argsort()[::-1].argsort()
            logger.info(f'f1_ranks for markov is {f1_ranks}')
            logger.info(f'ranks_by_metrics for markov is {ranks_by_metrics}')
            logger.info(f'rank_prauc for markov is {rank_prauc}')
            logger.info(f'rank_vus for markov is {rank_vus}')

            # Step 2: Call the enhanced Markov Chain method
            objective, markov_aggregated_rank = enhanced_markov_chain_rank_aggregator(*rank_arrays)
            logger.info(f'try this {f1[markov_aggregated_rank.astype(int)-1]}')
            markov_aggregated_rank_df = pd.DataFrame(f1[markov_aggregated_rank.astype(int)-1],index=model_names,columns=rank_column_name_list)
            markov_aggregated_rank_html = markov_aggregated_rank_df.to_html()
            result_html += ''' <fieldset>
                        <legend>markov_aggregated_rank </legend>'''
            result_html +=markov_aggregated_rank_html
            result_html += '''</fieldset>'''



            # end ###################################################
            #                            *                          #
            #                           * *                         #
            #                          * * *                        #
            #                           * *                         #
            #                            *                          #
            # start again ###########################################
            # copeland_rank_aggregator

            objective, copeland_rank = copeland_rank_aggregator(*rank_arrays)
            logger.info(f'try this {f1[copeland_rank.astype(int)-1]}')
            copeland_rank_df = pd.DataFrame(f1[copeland_rank.astype(int)-1],index=model_names,columns=rank_column_name_list)
            copeland_rank_html = copeland_rank_df.to_html()
            result_html += ''' <fieldset>
                        <legend>copeland_rank </legend>'''
            result_html +=copeland_rank_html
            result_html += '''</fieldset>'''



            # end ###################################################
            #                            *                          #
            #                           * *                         #
            #                          * * *                        #
            #                           * *                         #
            #                            *                          #
            # start again ###########################################
            # spearmans_footrule_aggregator



            objective, spearmans_footrule = spearmans_footrule_aggregator(*rank_arrays)
            logger.info(f'try this {f1[spearmans_footrule.astype(int) - 1]}')
            spearmans_footrule_df = pd.DataFrame(f1[spearmans_footrule.astype(int) - 1], index=model_names,
                                            columns=rank_column_name_list)
            spearmans_footrule_html = spearmans_footrule_df.to_html()
            result_html += ''' <fieldset>
                        <legend>spearmans_footrule </legend>'''
            result_html += spearmans_footrule_html
            result_html += '''</fieldset>'''

            # end ###################################################
            # Borda ranks
            borda_rank = f1[borda(ranks)[1].astype(int)]
            logger.info(f'borda_rank is {borda_rank}')
            df_borda_rank = pd.DataFrame(borda_rank,index=model_names,columns=rank_column_name_list)
            df_borda_rank_html = df_borda_rank.to_html()

            result_html += ''' <fieldset>
                                <legend>Models Borda Rank from F1 Matrix</legend>'''
            result_html += df_borda_rank_html
            result_html += '''</fieldset>'''

            # Trimmed Borda ranks
            trimmed_borda_rank = f1[trimmed_borda(ranks=ranks, aggregation_type='borda', metric='pagerank')[1].astype(int)]

            logger.info(f'trimmed_borda_rank is {trimmed_borda_rank}')

            df_trimmed_borda_rank = pd.DataFrame(trimmed_borda_rank,index=model_names,columns=rank_column_name_list)
            df_trimmed_borda_rank_html = df_trimmed_borda_rank.to_html()

            result_html += ''' <fieldset>
                                        <legend>Models Trimmed Borda Rank Matrix</legend>'''
            result_html += df_trimmed_borda_rank_html
            result_html += '''</fieldset>'''



            # Partial Borda ranks
            partial_borda_rank = f1[partial_borda(ranks, top_k=5)[1].astype(int)]
            logger.info(f'partial_borda_rank is {partial_borda_rank}')

            df_partial_borda_rank = pd.DataFrame(partial_borda_rank,index=model_names,columns=rank_column_name_list)
            df_partial_borda_rank_html = df_partial_borda_rank.to_html()

            result_html += ''' <fieldset>
                                                <legend>Models Partial Borda Rank Matrix</legend>'''
            result_html += df_partial_borda_rank_html
            result_html += '''</fieldset>'''

            # Trimmed partial Borda ranks
            trimmed_partial_borda_rank = f1[trimmed_partial_borda(ranks, top_k=5, metric='influence', aggregation_type='borda')[1].astype(int)]
            logger.info(f'trimmed_partial_borda_rank is {trimmed_partial_borda_rank}')

            df_trimmed_partial_borda_rank = pd.DataFrame(trimmed_partial_borda_rank, index=model_names,columns=rank_column_name_list)
            df_trimmed_partial_borda_rank_html = df_trimmed_partial_borda_rank.to_html()

            result_html += ''' <fieldset>
                                                        <legend>Models Trimmed Partial Borda Rank Matrix</legend>'''
            result_html += df_trimmed_partial_borda_rank_html
            result_html += '''</fieldset>'''



            top_k = 5
            reliability = influence(ranks, aggregation_type='borda', top_k=top_k)
            trimmed_ranks = ranks[np.argsort(-1 * reliability)[:1], :]
            borda_trimmed_ranks = (f1[borda(trimmed_ranks)[1].astype(int)])
            logger.info(f'borda_trimmed_ranks is {borda_trimmed_ranks}')

            df_borda_trimmed_ranks = pd.DataFrame(borda_trimmed_ranks, index=model_names,columns=rank_column_name_list)
            df_borda_trimmed_ranks_html = df_borda_trimmed_ranks.to_html()

            result_html += ''' <fieldset>
                                                                <legend>Models Borda Trimmed Rank Matrix</legend>'''
            result_html += df_borda_trimmed_ranks_html
            result_html += '''</fieldset>'''



            # Clustering EMPIRICAL INFLUENCE AND ROBUST RANK AGGREGATION
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit_predict(reliability.reshape((-1, 1)))
            cluster_ids, counts = np.unique(clustering, return_counts=True)
            largest_cluster_idx = cluster_ids[np.argmax(counts)]  # Largest cluster

            logger.info(f'Clustering:  {clustering}')
            clustering_df = pd.DataFrame(clustering,columns=['cluster'])
            clustering_df_html = clustering_df.to_html()


            result_html += ''' <fieldset>
            <legend>EMPIRICAL INFLUENCE AND ROBUST RANK AGGREGATION</legend>
            '''
            result_html += clustering_df_html
            result_html += '''</fieldset>'''


            most_reliable_cluster_idx = np.argmax([
                np.sum(reliability[np.where(clustering == 0)[0]]),
                np.sum(reliability[np.where(clustering == 1)[0]])])

            logger.info(f'Most reliable cluster idx: {most_reliable_cluster_idx}')
            logger.info(f'Largest: {largest_cluster_idx}')
            result_html += ''' <fieldset>
                               <p>Most reliable cluster idx: {},Largest {}</p>'''.format(most_reliable_cluster_idx,largest_cluster_idx)
            result_html += '''</fieldset>'''





            # conclusion
            from sklearn.metrics import ndcg_score
            cluster_ids, counts = np.unique(clustering, return_counts=True)
            max_cluster = cluster_ids[np.argmax(counts)]

            _, trimmed_kemeny_rank = kemeny(ranks_by_metrics[np.where(clustering == max_cluster)[0], :], verbose=False)
            trimmed_kemeny_rank = trimmed_kemeny_rank.astype(int)

            praucs = models_performance_matrix.iloc[:, 0].to_numpy().squeeze()
            f1s = models_performance_matrix.iloc[:, 1].to_numpy().squeeze()
            model_names = list(models_performance_matrix.index)

            logger.info('==== Statistics ====')
            logger.info(f'Rank by PR-AUC: {rank_prauc}')
            logger.info(f'Rank by F1: {rank_f1}')
            logger.info(f'Predicted rank(trimmed_kemeny_rank): {trimmed_kemeny_rank}')
            logger.info(f'Max PR-AUC: {np.max(praucs)} is achieved by {model_names[np.argmax(praucs)]}')
            logger.info(f'Max F-1: {np.max(f1s)} is achieved by {model_names[np.argmax(f1s)]}')
            logger.info(
                f'Our chosen model is {model_names[trimmed_kemeny_rank[0]]} which has PR-AUC={praucs[trimmed_kemeny_rank[0]]} and best F-1={f1s[trimmed_kemeny_rank[0]]}')
            logger.info(
                f'NDCG of predicted ranks with PR-AUC={ndcg_score(praucs.reshape((1, -1)), trimmed_kemeny_rank.reshape((1, -1)))} and best F-1={ndcg_score(f1s.reshape((1, -1)), trimmed_kemeny_rank.reshape((1, -1)))}')

            result_html += ''' <fieldset>
                                      <legend>Statistics</legend>'''


            rank_prauc_model_name_list = [model_names[int(i)] for i in rank_prauc]
            result_html += '''<p>Rank by PR-AUC : {}<p/>'''.format(rank_prauc_model_name_list)

            rank_f1_model_name_list = [model_names[int(i)] for i in rank_f1]
            result_html += '''<p>Rank by F1 : {}<p/>'''.format(rank_f1_model_name_list)

            trimmed_kemeny_rank_model_name_list = [model_names[int(i)] for i in trimmed_kemeny_rank]
            result_html += '''<p>Kemeny Predicted rank : {}<p/>'''.format(trimmed_kemeny_rank_model_name_list)

            result_html += '''<p>Max PR-AUC: {} is achieved by {}<p/>'''.format(np.max(praucs),model_names[np.argmax(praucs)])
            result_html += '''<p>Max F-1: {} is achieved by {}<p/>'''.format(np.max(f1s),model_names[np.argmax(f1s)])
            best_model = model_names[trimmed_kemeny_rank[0]]
            result_html += '''<p>Our chosen model is : {} which has PR-AUC (using Kemeny)= {} and best F-1= {}<p/>'''.format(best_model,praucs[trimmed_kemeny_rank[0]],f1s[trimmed_kemeny_rank[0]])
            result_html += '''<p>(Normalized Discounted Cumulative Gain) NDCG of predicted ranks with PR-AUC= {} and best F-1= {}<p/>'''.format(ndcg_score(praucs.reshape((1, -1)), trimmed_kemeny_rank.reshape((1, -1))),ndcg_score(f1s.reshape((1, -1)), trimmed_kemeny_rank.reshape((1, -1))))



            update_data_best_model_by_name(_data_name=dataset_entity,_best_model=best_model)

            result_html += '''</fieldset>'''
            return result_html
