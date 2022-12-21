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
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename='fixed_only',
    #      query_filename='fixed_only',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename='prev_code',
    #      query_filename='prev_code',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename='prev_code',
    #      query_filename='prev_code',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename=['buggy_only','commit_msg'],
    #      query_filename=['buggy_only','commit_msg'],
    #      k=5,
    #      concatenate=True,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='small',
    #      src_lang='en_XX', tgt_lang='en_XX',
    #      db_data_filename='commit_msg',
    #      query_filename='commit_msg',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename='commit_msg',
    #      query_filename='commit_msg',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_data_filename='commit_msg',
    #      query_filename='commit_msg',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang=None, tgt_lang=None,
    #      db_data_filename='fixed_only',
    #      query_filename='fixed_only',
    #      k=5,
    #      concatenate=False,
    #      method='tfidf'),

    # dict(dataset_size='small',
    #      src_lang=None, tgt_lang=None,
    #      db_data_filename='fixed_only',
    #      query_filename='fixed_only',
    #      k=1,
    #      concatenate=False,
    #      method='tfidf'),

    # dict(dataset_size='small',
    #      src_lang=None, tgt_lang=None,
    #      db_data_filename='prev_code',
    #      query_filename='prev_code',
    #      k=5,
    #      concatenate=False,
    #      method='tfidf'),
    #
    # dict(dataset_size='small',
    #      src_lang=None, tgt_lang=None,
    #      db_data_filename='prev_code',
    #      query_filename='prev_code',
    #      k=1,
    #      concatenate=False,
    #      method='tfidf')

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='train',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=2,  # ensure k = k+1 for query=train
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='eval',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='test',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='train',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=6,  # ensure k = k+1 for query=train
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='eval',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='test',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),

    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='train',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=3,  # ensure k = k+1 for query=train
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='eval',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=2,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='small',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='test',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=2,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='train',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=2,  # ensure k = k+1 for query=train
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='eval',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='test',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=1,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='train',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=3,  # ensure k = k+1 for query=train
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='eval',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=2,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='test',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=2,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='train',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=6,  # ensure k = k+1 for query=train
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='eval',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),
    #
    # dict(dataset_size='medium',
    #      src_lang='java', tgt_lang='java',
    #      db_path='train', query_path='test',
    #      db_data_filename='buggy_only',
    #      query_filename='buggy_only',
    #      k=5,
    #      concatenate=False,
    #      method='plbart'),

]

ks = [5] #1
db_path = 'train'
dataset_sizes = ['small']#, 'medium']
query_paths = ['train', 'eval', 'test']
filenames = [['prev_code', 'buggy_only'],
             ['prev_code','commit_msg'],
             ['buggy_only','commit_msg']
             ]
methods = ['plbart']
variations = []
for dataset_size in dataset_sizes:
    for query_path in query_paths:
        for filename in filenames:
            for k in ks:
                for method in methods:
                    variation = {
                        'dataset_size': dataset_size,
                        'src_lang': 'java' if method == 'plbart' else None,
                        'tgt_lang': 'java' if method == 'plbart' else None,
                        'db_path': db_path, 'query_path': query_path,
                        'db_data_filename': filename,
                        'query_filename': filename,
                        'k': k + 1 if 'train' == query_path else k,
                        'concatenate': True,
                        'method': method,
                    }
                    variations.append(variation)

print('Running ', len(variations), ' variations')
