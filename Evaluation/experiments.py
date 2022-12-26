variations = []

ks = [5]
db_path = 'train'
dataset_sizes = ['small']
query_paths = ['train', 'eval', 'test']
filenames = ['prev_code', 'buggy_only', 'commit_msg']
methods = ['modit']
model_ckpt = '../../../shared/modit_checkpoints/small.parent_contexed_commit.child_code/checkpoint_best.pt'
variations = []
for dataset_size in dataset_sizes:
    for query_path in query_paths:
        for filename in filenames:
            for k in ks:
                for method in methods:
                    variation = {
                        'dataset_size': dataset_size,
                        'src_lang': 'java' if method in ['plbart', 'modit'] else None,
                        'tgt_lang': 'java' if method in ['plbart', 'modit'] else None,
                        'db_path': db_path, 'query_path': query_path,
                        'db_data_filename': filename,
                        'query_filename': filename,
                        'k': k + 1 if 'train' == query_path else k,
                        'concatenate': False,
                        'method': method,
                        'model_ckpt': model_ckpt,
                    }
                    variations.append(variation)

print('Running ', len(variations), ' variations')
