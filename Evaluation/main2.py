import json
import time
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PLBartTokenizer, PLBartModel


def concatenate_data(data, token=' <b> '):
    return [x + token + y for x, y in zip(data[0], data[1])]


def create_gojo(tokenizer,
                model,
                train_set,
                embeddings_filename='buggy_commit.npy'
                ):
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
    # db_data_embeddings = create_gojo(tokenizer,
    #                                  model,
    #                                  concatenate_data(db_data) if concatenate else db_data
    #                                  )

    # query_embeddings = db_data_embeddings
    # query_embeddings = create_gojo(tokenizer,
    #                                  model,
    #                                  concatenate_data(queries) if concatenate else queries
    #                                  )

    # similarity_matrix = np.dot(db_data_embeddings, query_embeddings.T)

    # np.save(filename, similarity_matrix)
    similarity_matrix = np.load(filename)

    similarity_matrix = torch.from_numpy(similarity_matrix)
    top_k_similarity_matrix = torch.topk(similarity_matrix, k, dim=1)

    return top_k_similarity_matrix


def map_top_k_results(top_k_results,
                      queries,
                      results_filename,
                      concatenate,
                      ):

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


def func_name(tokenizer,
              model,
              db_data,
              queries=[],
              k=11,
              similarity_matrix_filename='similarity_matrix.npy',
              results_filename='results.json',
              concatenate=False,
              ):
    if not queries:
        queries = db_data

    start_time = time.time()
    top_k_similarity_matrix = get_top_k_similarity_matrix(tokenizer, model, db_data,
                                                          queries, k, similarity_matrix_filename,
                                                          concatenate)
    print('Obtained top k results', time.time() - start_time)

    return map_top_k_results(top_k_similarity_matrix,
                             queries,
                             results_filename,
                             concatenate,
                             )


def compare_embeddings(tokenizer,
                       model,
                       dataloader,
                       k):
    for x in dataloader:
        print(x)
        embeddings = model(x)
        print()


def load_model_and_tokenizer(device, tf_save_directory="uclanlp/plbart-base"):
    tokenizer = PLBartTokenizer.from_pretrained('uclanlp/plbart-base',
                                                src_lang="java", tgt_lang="java",
                                                )
    model = PLBartModel.from_pretrained('uclanlp/plbart-base')  # .to(device)
    model.eval()

    return tokenizer, model


if __name__ == '__main__':
    basepath = path.dirname(__file__)
    tokenized_input = path.abspath(path.join(basepath, "../Patch-Dataset/tokenized_input"))
    tf_save_directory = path.abspath(path.join(basepath, "../PLBART_model/"))
    code_embeddings_file_path = path.abspath(path.join(basepath, "../Patch-Dataset/code_embeddings.pkl"))
    code_ref_filepath = path.abspath(path.join(basepath, "../Patch-Dataset/small/train/data.prev_code"))
    
    buggy_filepath = path.abspath(path.join(basepath, "../Patch-Dataset/small/train/data.buggy_only"))
    commit_msg_filepath = path.abspath(path.join(basepath, "../Patch-Dataset/small/train/data.commit_msg"))
    fixed_only_filepath = path.abspath(path.join(basepath, "../Patch-Dataset/small/train/data.fixed_only"))
    next_code_filepath = path.abspath(path.join(basepath, "../Patch-Dataset/small/train/data.next_code"))
    prev_code_filepath = path.abspath(path.join(basepath, "../Patch-Dataset/small/train/data.prev_code"))

    score_file_path = path.abspath(path.join(basepath, "../Evaluation/top_k_results_2.json"))
    code_encodings_file_path = path.abspath(path.join(basepath, "../Patch-Dataset/code_encodings.pkl"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    plbart_tokenizer, plbart_model = load_model_and_tokenizer(device)

    # Prev_code
    # with open(code_ref_filepath, 'r') as f:
    #     db_data = f.readlines()

    # func_name(plbart_tokenizer, plbart_model, db_data, [], 11,  
    #           'similarity_matrix_prev.npy', 'results_prev.json', False)

    # # buggy+nl_desc
    db_data = []
    with open(buggy_filepath, 'r') as f:
        db_data.append(f.readlines())
    with open(commit_msg_filepath, 'r') as f:
        db_data.append(f.readlines())
        
        
    func_name(plbart_tokenizer, plbart_model, db_data, [], 11,
              'similarity_matrix_buggy_desc.npy',
              'results_buggy_desc.json', True)