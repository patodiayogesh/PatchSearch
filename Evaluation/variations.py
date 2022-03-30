import time
from os import path
from os.path import exists

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
from transformers import PLBartTokenizer, PLBartModel

func_arguments = {}


def concatenate_data(data, token=' <b> '):
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


def calculate_edit_distance(top_k_results,
                            queries,
                            concatenate,
                            fixed_only_filepath
                            ):
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

    edit_distance_filename = 'edit_distances' + create_filename_with_k()
    with open(edit_distance_filename, 'w') as f:
        for elem in edit_distances:
            f.write(str(elem) + "\n")

    visualization_filename = 'visualization' + create_filename_with_k()
    sns.distplot(edit_distances)
    plt.savefig(visualization_filename)
    plt.show()



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

    plbart_tokenizer, plbart_model = load_model_and_tokenizer(device, src_lang, tgt_lang)

    if concatenate:
        db_data = []
        with open(filepaths[db_data_filename[0]], 'r') as f:
            db_data.append(f.readlines())
        with open(filepaths[db_data_filename[1]], 'r') as f:
            db_data.append(f.readlines())

        db_data[0] = db_data[0][:100]
        db_data[1] = db_data[1][:100]

    else:
        with open(filepaths[db_data_filename], 'r') as f:
            db_data = f.readlines()
        db_data = db_data[:100]

    compute_similarity_matrix_and_edit_dist(plbart_tokenizer, plbart_model,
                                            db_data, [], k,
                                            concatenate,
                                            filepaths['fixed_only'])


def main():
    variations = [

        dict(dataset_size='small',
             src_lang='java', tgt_lang='java',
             db_data_filename='prev_code',
             k=2,
             concatenate=False),

        dict(dataset_size='small',
             src_lang='java', tgt_lang='java',
             db_data_filename=['buggy_only','commit_msg'],
             k=2,
             concatenate=True),
    ]
    for variation in variations:
        global func_arguments
        func_arguments = variation
        evaluate(**func_arguments)

if __name__ == '__main__':
    main()
