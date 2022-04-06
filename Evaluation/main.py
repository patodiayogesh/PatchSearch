import time
from os import path

import torch

from tfidf import TfIdfEvaluator
from plbart import PlBartEvaluator

func_arguments = {}


def set_filepaths(obj, basepath, dataset_size):
    """
    Function to set filepaths in evaluator object.

    :param obj: Evaluator object
    :param basepath: Path Directory
    :param dataset_size: small or large dataset
    :return: None

    """
    db_filepaths = dict()
    db_filepaths['buggy_only'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.buggy_only"))
    db_filepaths['commit_msg'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.commit_msg"))
    db_filepaths['fixed_only'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.fixed_only"))
    db_filepaths['next_code'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.next_code"))
    db_filepaths['prev_code'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.prev_code"))
    obj.db_filepaths = db_filepaths

    query_filepaths = dict()
    query_filepaths['buggy_only'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/test/data.buggy_only"))
    query_filepaths['commit_msg'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/test/data.commit_msg"))
    query_filepaths['fixed_only'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/test/data.fixed_only"))
    query_filepaths['next_code'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/test/data.next_code"))
    query_filepaths['prev_code'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/test/data.prev_code"))
    obj.query_filepaths = query_filepaths

    return


def compute_similarity_matrix_and_edit_dist(obj, method):
    """
    Function to compute similarity matrix using evaluator obj method.
    Calls objects edit distance method to calculate edit distance
    and plot distribution.

    :param obj: Evaluator Object
    :param method: plbart or tfidf
    :return: None
    """

    if not obj.queries:
        obj.queries = obj.db_data

    similarity_matrix_filename = 'similarity_matrix' + obj.create_filename() + '_' + method + '.npy'
    start_time = time.time()
    top_k_similarity_matrix = obj.get_top_k_similarity_matrix(similarity_matrix_filename)
    print('Obtained top k results', time.time() - start_time)
    # return map_top_k_results(top_k_similarity_matrix,
    #                          queries,
    #                          results_filename,
    #                          concatenate,
    #                          )
    obj.calculate_edit_distance(top_k_similarity_matrix)

    return None


def evaluate(dataset_size,
             src_lang, tgt_lang,
             db_data_filename,
             query_filename,
             k,
             concatenate,
             method
             ):
    """
    Create Evaluator Object
    Set File paths to read from
    Load Queries and DB
    Find Top-K similar results from DB for each Query

    :param dataset_size: directory from which to load datasets from
    :param src_lang: plbart source language
    :param tgt_lang: plbart target language
    :param db_data_filename: db filename to use for comparison
    :param query_filename: query filename to use for comparison
    :param k: denote the top-k similar data from db for each query
    :param concatenate: if reading from multiple data sources and concatenation is required
    :param method: plbart or tfidf
    :return: None
    """
    if method == 'plbart':
        evaluator_obj = PlBartEvaluator(dataset_size,
                                        src_lang, tgt_lang,
                                        db_data_filename,
                                        query_filename,
                                        k,
                                        concatenate)

    elif method == 'tfidf':
        evaluator_obj = TfIdfEvaluator(dataset_size,
                                       db_data_filename,
                                       query_filename,
                                       k,
                                       concatenate)
    else:
        print('Evaluator Method Undefined ', method)
        return None

    set_filepaths(evaluator_obj, path.dirname(__file__), dataset_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evaluator_obj.device = device

    # Read Data from DB and Queries
    if concatenate:
        db_data = []
        # print("let us see\n")
        # print(evaluator_obj.db_filepaths[db_data_filename[0]])
        with open(evaluator_obj.db_filepaths[db_data_filename[0]], 'r') as f:
            db_data.append(f.readlines())
        with open(evaluator_obj.db_filepaths[db_data_filename[1]], 'r') as f:
            db_data.append(f.readlines())

        queries = []
        with open(evaluator_obj.query_filepaths[query_filename[0]], 'r') as f:
            queries.append(f.readlines())
        with open(evaluator_obj.query_filepaths[query_filename[1]], 'r') as f:
            queries.append(f.readlines())

        # db_data[0] = db_data[0][:100]
        # db_data[1] = db_data[1][:100]

    else:
        with open(evaluator_obj.db_filepaths[db_data_filename], 'r') as f:
            db_data = f.readlines()

        with open(evaluator_obj.query_filepaths[query_filename], 'r') as f:
            queries = f.readlines()

        db_data = db_data[:100]
        queries = queries[:10]

    evaluator_obj.db_data = db_data
    evaluator_obj.queries = queries

    compute_similarity_matrix_and_edit_dist(evaluator_obj, method)

    return None


def main():
    """
    Variations for testing
    Saves all variation results
    :return: None
    """
    variations = [

        # dict(dataset_size='small',
        #      src_lang='java', tgt_lang='java',
        #      db_data_filename='prev_code',
        #      k=11,
        #      concatenate=False),

        # dict(dataset_size='small',
        #      src_lang='java', tgt_lang='java',
        #      db_data_filename=['buggy_only','commit_msg'],
        #      k=11,
        #      concatenate=True),

        # dict(dataset_size='small',
        #      src_lang='java', tgt_lang='java',
        #      db_data_filename='prev_code',
        #      k=2,
        #      concatenate=False),

        # dict(dataset_size='small',
        #      src_lang='java', tgt_lang='java',
        #      db_data_filename=['buggy_only','commit_msg'],
        #      k=2,
        #      concatenate=True),

        # dict(dataset_size='small',
        #      src_lang='java', tgt_lang='java',
        #      db_data_filename='fixed_only',
        #      query_filename='fixed_only',
        #      k=5,
        #      concatenate=False,
        #      method='tfidf'),

        dict(dataset_size='small',
             src_lang=None, tgt_lang=None,
             db_data_filename='fixed_only',
             query_filename='fixed_only',
             k=5,
             concatenate=False,
             method='plbart'),

    ]

    for variation in variations:
        global func_arguments
        func_arguments = variation
        evaluate(**func_arguments)


if __name__ == '__main__':
    main()
