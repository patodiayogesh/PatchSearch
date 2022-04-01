import time
from os import path
from os.path import exists
import json

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
from transformers import PLBartTokenizer, PLBartModel

func_arguments = {}

def concatenate_data(data, token=' <b> '):
    # Concatenate data when rading from multiple sources
    return [x + token + y for x, y in zip(data[0], data[1])]


def create_filename_with_k():
    global func_arguments
    if not func_arguments['concatenate']:
        return '_' + func_arguments['dataset_size'] \
               + '_' + func_arguments['src_lang'] \
               + '_' + func_arguments['tgt_lang'] \
               + '_' + func_arguments['db_data_filename'] \
               + '_' + str(func_arguments['k'])

    else:
        db_data_filename = '_'.join(func_arguments['db_data_filename'])
        return '_' + func_arguments['dataset_size'] \
               + '_' + func_arguments['src_lang'] \
               + '_' + func_arguments['tgt_lang'] \
               + '_' + db_data_filename \
               + '_' + str(func_arguments['k'])

def create_filename():
    global func_arguments
    if not func_arguments['concatenate']:
        return '_' + func_arguments['dataset_size'] \
               + '_' + func_arguments['src_lang'] \
               + '_' + func_arguments['tgt_lang'] \
               + '_' + func_arguments['db_data_filename']

    else:
        db_data_filename = '_'.join(func_arguments['db_data_filename'])
        return '_' + func_arguments['dataset_size'] \
               + '_' + func_arguments['src_lang'] \
               + '_' + func_arguments['tgt_lang'] \
               + '_' + db_data_filename


def create_load_embeddings(tokenizer,
                           model,
                           train_set,
                           embeddings_filename
                           ):
    '''
    Function to create or load data embeddings
    '''
    if exists(embeddings_filename):
        print(embeddings_filename +' exists')
        embeddings = np.load(embeddings_filename)
        return embeddings

    embeddings = []
    for j in range(len(train_set)):
        print(j)
        code_encodings = tokenizer(train_set[j],
                                   return_tensors="pt")  # .to(device)
        with torch.no_grad():
            code_embeddings = model(**code_encodings)
        code_embeddings = torch.mean(code_embeddings.last_hidden_state, dim=1)
        code_embeddings = code_embeddings.flatten()

        embeddings.append(code_embeddings.detach().cpu().numpy())

    embeddings = np.array(embeddings)
    np.save(embeddings_filename, embeddings)
    return embeddings


def get_top_k_similarity_matrix(tokenizer,
                                model,
                                db_data,
                                queries,
                                k,
                                filename,
                                concatenate=False,
                                ):
    '''
    Compute similarity matrix between db and query embeddings
    Similarity computed using Levenshtein distance
    Return top-k results
    '''
    if exists(filename):
        print(filename+' exists')
        similarity_matrix = np.load(filename)
    else:

        embeddings_filename = 'embeddings_db_data' + create_filename() + '.npy'
        db_data_embeddings = create_load_embeddings(tokenizer,
                                                    model,
                                                    concatenate_data(db_data) if concatenate else db_data,
                                                    embeddings_filename
                                                    )

        query_embeddings = db_data_embeddings
        # query_embeddings = create_load_embeddings(tokenizer,
        #                                           model,
        #                                           concatenate_data(queries) if concatenate else queries,
        #                                           embeddings_filename
        #                                           )

        similarity_matrix = np.dot(db_data_embeddings, query_embeddings.T)
        np.save(filename, similarity_matrix)

    similarity_matrix = torch.from_numpy(similarity_matrix)
    top_k_similarity_matrix = torch.topk(similarity_matrix, k, dim=1)

    return top_k_similarity_matrix

def map_top_k_results(top_k_results,
                      queries,
                      results_filename,
                      concatenate,
                      ):
    '''
    Function to map fixed code, buggy code, commit message, prev code
    for queries
    '''

    with open(code_ref_filepath, 'r') as f:
        prev_code = f.readlines()
    with open(buggy_filepath, 'r') as f:
        buggy_data = f.readlines()
    with open(commit_msg_filepath, 'r') as f:
        commit_msg_data = f.readlines()
    with open(fixed_only_filepath, 'r') as f:
        fixed_only_data = f.readlines()
    with open(next_code_filepath, 'r') as f:
        next_code_data = f.readlines()

    start_time = time.time()
    top_k_indices = top_k_results.indices.data
    top_k_scores = top_k_results.values.data
    results = []

    for i, query in enumerate(queries if not concatenate else queries[0]):

        d = []
        indices = top_k_indices[i]
        scores = top_k_scores[i]
        for index, score in zip(indices, scores):
            if index != i:
                d.append({'prev_code': prev_code[index],
                          'score': score.cpu().detach().item(),
                          'buggy_code': buggy_data[index],
                          'commit_msg': commit_msg_data[index],
                          'fixed_code': fixed_only_data[index],
                          'next_code': next_code_data[index],
                          })

        results.append([{'buggy_code': queries[0][i], 'commit_msg': queries[1][i]} if concatenate else query,
                        d])

    with open(results_filename, 'w') as f:
        f.write(json.dumps(results))

    print('Saved top k results', time.time() - start_time)
    return results


def calculate_edit_distance(top_k_results,
                            queries,
                            concatenate,
                            fixed_only_filepath
                            ):
    
    # Compute Normalized Levenshtein distance for top-k results
    # Plot distance distribution
    edit_distance_filename = 'edit_distances' + create_filename_with_k()
    
    if exists(edit_distance_filename):
        print(edit_distance_filename, ' exists')
        edit_distances = []
        f = open(edit_distance_filename,'r') 
        for line in f:
            edit_distances.append(line.strip('\n'))
        f.close()
    else:
        with open(fixed_only_filepath, 'r') as f:
            fixed_only_data = f.readlines()

        start_time = time.time()
        top_k_indices = top_k_results.indices.data
        top_k_scores = top_k_results.values.data
        results = []

        edit_distances = []

        for i, query in enumerate(queries if not concatenate else queries[0]):

            d = []
            indices = top_k_indices[i]
            scores = top_k_scores[i]
            for index, score in zip(indices, scores):
                if index != i:
                    print(index, i)
                    levenshtein_dist = nltk.edit_distance(fixed_only_data[i], fixed_only_data[index])
                    normalized_levenshtein_dist = levenshtein_dist / (
                        max(len(fixed_only_data[i]), len(fixed_only_data[index])))
                    edit_distances.append(normalized_levenshtein_dist)

        with open(edit_distance_filename, 'w') as f:
            for elem in edit_distances:
                f.write(str(elem) + "\n")

    visualization_filename = 'visualization' + create_filename_with_k()
    sns.distplot(edit_distances)
    plt.savefig(visualization_filename)

    return


def compute_similarity_matrix_and_edit_dist(tokenizer,
                                            model,
                                            db_data, queries,
                                            k,
                                            concatenate,
                                            fixed_only_filepath
                                            ):
    if not queries:
        queries = db_data

    similarity_matrix_filename = 'similarity_matrix'+create_filename()+'.npy'
    start_time = time.time()
    top_k_similarity_matrix = get_top_k_similarity_matrix(tokenizer, model, db_data,
                                                          queries, k, similarity_matrix_filename,
                                                          concatenate)
    print('Obtained top k results', time.time() - start_time)

    # return map_top_k_results(top_k_similarity_matrix,
    #                          queries,
    #                          results_filename,
    #                          concatenate,
    #                          )

    return calculate_edit_distance(top_k_similarity_matrix,
                                   queries,
                                   concatenate,
                                   fixed_only_filepath
                                   )

def load_model_and_tokenizer(device, source_lang, target_lang,
                             tf_save_directory="uclanlp/plbart-base"
                             ):
    tokenizer = PLBartTokenizer.from_pretrained(tf_save_directory,
                                                src_lang=source_lang,
                                                tgt_lang=target_lang,
                                                )
    model = PLBartModel.from_pretrained('uclanlp/plbart-base')  # .to(device)
    model.eval()

    return tokenizer, model


def evaluate(dataset_size,
             src_lang, tgt_lang,
             db_data_filename,
             k,
             concatenate
             ):

    basepath = path.dirname(__file__)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    filepaths = {}
    filepaths['buggy_only'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.buggy_only"))
    filepaths['commit_msg'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.commit_msg"))
    filepaths['fixed_only'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.fixed_only"))
    filepaths['next_code'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.next_code"))
    filepaths['prev_code'] = path.abspath(
        path.join(basepath, "../Patch-Dataset/" + dataset_size + "/train/data.prev_code"))

    # Create PLBART model and tokenizer
    plbart_tokenizer, plbart_model = load_model_and_tokenizer(device, src_lang, tgt_lang)

    if concatenate:
        db_data = []
        with open(filepaths[db_data_filename[0]], 'r') as f:
            db_data.append(f.readlines())
        with open(filepaths[db_data_filename[1]], 'r') as f:
            db_data.append(f.readlines())

        # db_data[0] = db_data[0][:100]
        # db_data[1] = db_data[1][:100]

    else:
        with open(filepaths[db_data_filename], 'r') as f:
            db_data = f.readlines()
        # db_data = db_data[:100]

    # Compute the similarity matrix and edit distance
    compute_similarity_matrix_and_edit_dist(plbart_tokenizer, plbart_model,
                                            db_data, [], k,
                                            concatenate,
                                            filepaths['fixed_only'])
    
    del plbart_tokenizer
    del plbart_model
    
    return

def main():
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

    ]

    for variation in variations:
        global func_arguments
        func_arguments = variation
        evaluate(**func_arguments)
        #gc.collect()

if __name__ == '__main__':
    main()
