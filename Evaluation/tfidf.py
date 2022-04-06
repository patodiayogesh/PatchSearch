from sklearn.feature_extraction.text import TfidfVectorizer
from utils import Evaluator
from os.path import exists

import torch
import numpy as np


class TfIdfEvaluator(Evaluator):

    def __init__(self, dataset_size, src_lang, tgt_lang, db_data_filename, query_filename, k, concatenate):
        super().__init__(dataset_size, src_lang, tgt_lang, db_data_filename, query_filename, k, concatenate)

    def get_top_k_similarity_matrix(self,
                                    filename):

        """
        Compute similarity matrix between db and query tfidf vectors
        Return top-k results

        :param filename: similarity matrix filename
        :return : torch top-k results
        """
        if exists(filename):
            print(filename + ' exists')
            similarity_matrix = np.load(filename)

        else:
            # Tfidf Vectorizer to fit on db data
            vectorizer = TfidfVectorizer()
            db_data_vectors = vectorizer.fit_transform(self.db_data)
            query_vectors = vectorizer.transform(self.queries)

            # Get np array format of tfidf scores
            db_data_matrix = np.array(db_data_vectors.todense())
            query_matrix = np.array(query_vectors.todense())

            # print(db_data_matrix)
            # row = db_data_matrix[10]
            # sns.distplot(row)
            # plt.show()

            similarity_matrix = np.dot(db_data_matrix, query_matrix.T)
            np.save(filename, similarity_matrix)

        # Get top-k results
        similarity_matrix = torch.from_numpy(similarity_matrix)
        top_k_similarity_matrix = torch.topk(similarity_matrix, self.k, dim=1)

        self.top_k_similarity_matrix = top_k_similarity_matrix
        return top_k_similarity_matrix

    def calculate_edit_distance(self, top_k_similarity_matrix):
        """
        Call Parent class function and pass specific filename

        :param top_k_similarity_matrix: top-k results from get_top_k_similarity_matrix
        :return: None
        """
        edit_distance_filename = 'edit_distances' + self.create_filename_with_k() + '_tfidf'
        visualization_filename = 'visualization' + self.create_filename_with_k() + '_tfidf'
        super().calculate_edit_distance(top_k_similarity_matrix,
                                        edit_distance_filename,
                                        visualization_filename)
        return None
