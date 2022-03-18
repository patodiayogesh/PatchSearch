import heapq
import json
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from fairseq.models.bart import BARTModel
from fairseq.data import data_utils
from transformers import PLBartTokenizer, PLBartModel
from torch.utils.data import Dataset, DataLoader, TensorDataset


def create_encodings(tokenizer, *model):
    with open(tokenized_input, 'r') as f:
        tokens = f.readlines()

    code_encoding = []
    for token in tokens:
        token += ' <java>'
        code_encoding.append(tokenizer(token, return_tensors="pt"))

    f = open(code_encodings_file_path, 'wb')
    pickle.dump(code_encoding, f)
    f.close()

    return None


def checkpoint_load():


    model = BARTModel.from_pretrained(
        tf_save_directory,
        checkpoint_file="checkpoint_11_100000.pt",
        lang_dict='lang_dict.txt',
    )
    model.eval()

    with open(tokenized_input, 'r') as f:
        tokens = f.readlines()

    # query = tokens[0]+' <java>'
    # query = model.encode(query)

    query = 'public java.lang.String getTotalEndDate () { return new java.text.SimpleDateFormat ( "yyyy-MM-dd" ) . format ( endDates [ ( ( endDates.length ) - 1 ) ] . getTime () ) ; } <java>'
    query = model.encode(query)
    # answers = tokens[1:30]
    # batch_input = [query.split(' ') + answer.split(' ') for answer in answers]

    with torch.no_grad():
        # batch_input = data_utils.collate_tokens(
        #     batch_input, model.model.encoder.dictionary.pad(), left_pad=False
        # )
        prediction = model.extract_features(query)


class Query_DB_Dataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataset(query_texts, db_texts):
    dataset = [[query_texts[i], db_texts[j]]
               for i in range(len(query_texts))
               for j in range(len(db_texts))
               ]

    return dataset


def get_token_encodings(tokenizer, data):
    encodings = []
    for x in data:
        encodings.append(tokenizer(x,
                                   padding=True,
                                   return_tensors="pt")
                         )

    return Query_DB_Dataset(encodings)


def get_top_k_results(tokenizer, model, queries=[], k=10):
    with open(code_ref_filepath, 'r') as f:
        db_data = f.readlines()

    db_data = db_data[:100]

    if not queries:
        queries = db_data
    create_gojo(tokenizer,
                model,
                db_data)

    top_k_results = []

    for i in range(len(queries)):
        print('query: ', i)
        count = 0
        # query_db = create_dataset([queries[i]], db_data)
        # encodings = get_token_encodings(tokenizer, query_db)
        # dataloader = DataLoader(encodings, batch_size=64, shuffle=False, collate_fn= lambda x: x)

        # compare_embeddings(tokenizer,
        #                    model,
        #                    dataloader,
        #                    k,
        #                    )

        scores = compare_embeddings_for_loop(tokenizer,
                                             model,
                                             queries[i],
                                             db_data,
                                             k)

        top_k_results.append(dict(query=queries[i],
                                  similar_code=[dict(code=db_data[j],
                                                     score=s.item())
                                                for s, j in scores]))

    with open(score_file_path, 'w') as f:
        json.dump(top_k_results, f)


def compare_embeddings_for_loop(tokenizer,
                                model,
                                query,
                                db_data,
                                k):
    scores = []
    count = 0
    for j in range(len(db_data)):

        print(j)
        code_encodings = tokenizer([query, db_data[j]],
                                   padding=True,
                                   return_tensors="pt")  # .to(device)
        with torch.no_grad():
            embeddings = model(**code_encodings)
        code_embeddings_1 = torch.mean(embeddings.last_hidden_state[0], dim=1)
        code_embeddings_1 = code_embeddings_1.reshape(1, code_embeddings_1.shape[0])
        code_embeddings_2 = torch.mean(embeddings.last_hidden_state[1], dim=1)
        code_embeddings_2 = code_embeddings_2.reshape(1, code_embeddings_2.shape[0])

        score = F.cosine_similarity(code_embeddings_1, code_embeddings_2)

        if count < k:
            scores.append((score, j))
        elif count == k:
            scores.append((score, j))
            heapq.heapify(scores)
        else:
            heapq.heapreplace(scores, (score, j))

        count += 1

    return sorted(scores)

def create_gojo(tokenizer,
                model,
                train_set,
                ):


    for j in range(len(train_set)):

        print(j)
        code_encodings = tokenizer(train_set[j],
                                   return_tensors="pt")  # .to(device)
        with torch.no_grad():
            embeddings = model(**code_encodings)
        code_embeddings_1 = torch.mean(embeddings.last_hidden_state, dim=0)
        code_embeddings_1 = code_embeddings_1.reshape(1, code_embeddings_1.shape[0])
        code_embeddings_2 = torch.mean(embeddings.last_hidden_state[1], dim=1)
        code_embeddings_2 = code_embeddings_2.reshape(1, code_embeddings_2.shape[0])

        score = F.cosine_similarity(code_embeddings_1, code_embeddings_2)


    return sorted(scores)

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
    score_file_path = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Evaluation/top_k_results_2.json'
    code_encodings_file_path = '/Users/neelampatodia/Desktop/Yogesh/PatchSearch/Patch-Dataset/code_encodings.pkl'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    plbart_tokenizer, plbart_model = load_model_and_tokenizer(device)
    get_top_k_results(plbart_tokenizer, plbart_model)
    # compare_embeddings(plbart_tokenizer, plbart_model)
