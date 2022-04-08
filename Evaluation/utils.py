import matplotlib.pyplot as plt
import nltk
from os.path import exists
import seaborn as sns


def concatenate_data(data, token=' <b> '):
    """
    Concatenate data when reading from multiple sources

    :param data: list of data content
    :param token: token to be append after each source
    :return: list of strings
    """

    return [x + token + y for x, y in zip(data[0], data[1])]


class Evaluator:

    """
    Class to compute similarity matrix for each type of evaluator
    Calculate edit distance for top-k results
    """

    def __init__(self, dataset_size, src_lang, tgt_lang, db_data_filename, query_filename, k, concatenate):

        self.dataset_size = dataset_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.db_data_filename = db_data_filename
        self.query_filename = query_filename
        self.k = k
        self.concatenate = concatenate

        self.db_data = None
        self.queries = None
        self.device = None
        self.query_filepaths = None
        self.db_filepaths = None
        self.top_k_similarity_matrix = None

    def get_top_k_similarity_matrix(self, filename):
        return None

    def calculate_edit_distance(self,
                                top_k_results,
                                edit_distance_filename,
                                visualization_filename,
                                ):
        """
        Compute Normalized Levenshtein distance for top-k results
        Plot distance distribution

        :param top_k_results: torch top-k result
        :param edit_distance_filename: filename to save edit distance
        :param visualization_filename: filename to save distribution plot
        :return: None
        """

        db_fixed_only_filepath = self.db_filepaths['fixed_only']
        query_fixed_only_filepath = self.query_filepaths['fixed_only']

        if exists(edit_distance_filename):
            print(edit_distance_filename, ' exists')
            edit_distances = []
            f = open(edit_distance_filename, 'r')
            for line in f:
                edit_distances.append(line.strip('\n'))
            f.close()
        else:
            with open(db_fixed_only_filepath, 'r') as f:
                db_fixed_only_data = f.readlines()
            with open(query_fixed_only_filepath, 'r') as f:
                query_fixed_only_data = f.readlines()

            top_k_indices = top_k_results.indices.data
            top_k_scores = top_k_results.values.data

            edit_distances = []
            # Compute Normalized Levenshtein distance for each query and top-k db results
            for i, _ in enumerate(self.queries if not self.concatenate else self.queries[0]):

                indices = top_k_indices[i]
                scores = top_k_scores[i]
                temp_edit_distances = []
                for index, _ in zip(indices, scores):
                    # if index != i:
                    print(index, i)
                    levenshtein_dist = nltk.edit_distance(query_fixed_only_data[i], db_fixed_only_data[index])
                    normalized_levenshtein_dist = levenshtein_dist / (
                        max(len(query_fixed_only_data[i]), len(db_fixed_only_data[index])))
                    temp_edit_distances.append(normalized_levenshtein_dist)
                edit_distances.append(min(temp_edit_distances))

            with open(edit_distance_filename, 'w') as f:
                for elem in edit_distances:
                    f.write(str(elem) + "\n")

        # Plot and save graph
        sns.distplot(edit_distances)
        plt.savefig(visualization_filename)
        plt.clf()

        return None

    def create_filename_with_k(self):
        """
        Filename to save result
        :return: str
        """
        if not self.concatenate:
            return '_' + self.dataset_size \
                   + '_' + str(self.src_lang) \
                   + '_' + str(self.tgt_lang) \
                   + '_' + self.db_data_filename \
                   + '_' + self.query_filename \
                   + '_' + str(self.k)

        else:
            db_data_filename = '_'.join(self.db_data_filename)
            query_filename = '_'.join(self.query_filename)
            return '_' + self.dataset_size \
                   + '_' + str(self.src_lang) \
                   + '_' + str(self.tgt_lang) \
                   + '_' + db_data_filename \
                   + '_' + query_filename \
                   + '_' + str(self.k)

    def create_filename(self):
        """
        Filename to save result
        :return: str
        """
        if not self.concatenate:
            return '_' + self.dataset_size \
                   + '_' + str(self.src_lang) \
                   + '_' + str(self.tgt_lang) \
                   + '_' + self.db_data_filename \
                   + '_' + self.query_filename

        else:
            db_data_filename = '_'.join(self.db_data_filename)
            query_filename = '_'.join(self.query_filename)
            return '_' + self.dataset_size \
                   + '_' + str(self.src_lang) \
                   + '_' + str(self.tgt_lang) \
                   + '_' + db_data_filename \
                   + '_' + query_filename
