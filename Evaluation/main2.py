import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PLBartTokenizer, PLBartModel


def get_top_k_results(tokenizer, model, db_data, queries, k):


    db_data_embeddings = create_gojo(tokenizer,
                                     model,
                                     db_data
                                    )

    query_embeddings = create_gojo(tokenizer,
                                     model,
                                     queries
                                     )

    similarity_matrix = np.dot(db_data_embeddings,query_embeddings.T)

    np.save('similarity_matrix', similarity_matrix)
    similarity_matrix = np.load('similarity_matrix.npy')

    similarity_matrix = torch.from_numpy(similarity_matrix)
    top_k_similarity_matrix = torch.topk(similarity_matrix, k, dim=1)
    # top_k_indices = top_k_similarity_matrix

    return top_k_similarity_matrix


def func_name(tokenizer, model, queries=[]):

    with open(code_ref_filepath, 'r') as f:
        db_data = f.readlines()

    with open(buggy_filepath, 'r') as f:
        buggy_data = f.readlines()
    with open(commit_msg_filepath, 'r') as f:
        commit_msg_data = f.readlines()
    with open(fixed_only_filepath, 'r') as f:
        fixed_only_data = f.readlines()
    with open(next_code_filepath, 'r') as f:
        next_code_data = f.readlines()



    #db_data = db_data[:10]

    if not queries:
        queries = db_data

    start_time = time.time()
    top_k_results = get_top_k_results(tokenizer, model, db_data, queries, k=10)
    print('Obtained top k results', time.time()-start_time)

    start_time = time.time()
    top_k_indices = top_k_results.indices.data
    top_k_scores = top_k_results.values.data
    results = []

    for i, query in enumerate(queries):

        d = []
        indices = top_k_indices[i]
        scores = top_k_scores[i]
        for index, score in zip(indices, scores):
            if index != i:
                d.append({'prev_code':db_data[index],
                          'score':score.cpu().detach().item(),
                          'buggy_code':buggy_data[index],
                          'commit_msg':commit_msg_data[index],
                          'fixed_code':fixed_only_data[index],
                          'next_code':next_code_data[index],
                          })

        results.append([query, d])

    with open('results.json', 'w') as f:
        f.write(json.dumps(results))

    print('Saved top k results', time.time() - start_time)
    return results



def create_gojo(tokenizer,
                model,
                train_set,
                ):
    embeddings = []
    for j in range(len(train_set)):
        code_encodings = tokenizer(train_set[j],
                                   return_tensors="pt")  # .to(device)
        with torch.no_grad():
            code_embeddings = model(**code_encodings)
        code_embeddings = torch.mean(code_embeddings.last_hidden_state, dim=1)
        code_embeddings = code_embeddings.flatten()

        embeddings.append(code_embeddings.detach().cpu().numpy())

    embeddings = np.array(embeddings)
    return embeddings


def compare_embeddings(tokenizer,
                       model,
                       dataloader,
                       k):
    for x in dataloader:
        print(x)
        embeddings = model(x)
        print()


def load_model_and_tokenizer(device, tf_save_directory="uclanlp/plbart-base"):
    tokenizer = PLBartTokenizer.from_pretrained(tf_save_directory, src_lang="java", tgt_lang="java")
    model = PLBartModel.from_pretrained(tf_save_directory)  # .to(device)
    model.eval()

    return tokenizer, model


def combined_input(plbart_tokenizer, plbart_model):
    code_encodings = plbart_tokenizer([query_1, query_2],
                                      padding=True,
                                      return_tensors="pt")

    outputs = plbart_model(**code_encodings)
    code_encodings_1 = outputs.last_hidden_state[0]
    code_encodings_2 = outputs.last_hidden_state[1]

    score = F.cosine_similarity(code_encodings_1, code_encodings_2)

    print(score)
    print(outputs)


if __name__ == '__main__':
    tokenized_input = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/tokenized_input'
    tf_save_directory = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/PLBART_model/'
    code_embeddings_file_path = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/code_embeddings.pkl'
    code_ref_filepath = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/small/train/data.prev_code'

    buggy_filepath = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/small/train/data.buggy_only'
    commit_msg_filepath = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/small/train/data.commit_msg'
    fixed_only_filepath = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/small/train/data.fixed_only'
    next_code_filepath = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/small/train/data.next_code'
    prev_code_filepath = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/small/train/data.prev_code'


    score_file_path = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Evaluation/top_k_results_2.json'
    code_encodings_file_path = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/code_encodings.pkl'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    plbart_tokenizer, plbart_model = load_model_and_tokenizer(device)

    func_name(plbart_tokenizer, plbart_model)
