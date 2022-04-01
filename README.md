# PatchSearch
AI4SE Patch Search project

Our project concerns itself with patch search, for which we use a patch dataset. We use PLBart to tokenize and encode the dataset.

Directory Structure:

The top directory includes the following folders: Evaluation and Patch-Dataset. Because the files inside Patch-Dataset are too large, we were not able to push it to git. However, they exist in the Patch Dataset Google Drive that is shared with you. 

How to run:

Go inside the Evaluation folder. Create a virtual environment with python 3.8 and download the packages inside requirements.txt 

Afterwards, run main.py inside the Evaluations folder.
Different variations can be tested by interacting with the variation variable
dataset_size= 'small', 'large'
src_lang='java','en_XX' 
tgt_lang='java','en_XX' 
db_data_filename= 'prev_code', 'buggy_code', 'commit_msg'. (Can read from 2 sources combined. Provide the values in a list)
k= ::int,
concatenate=True, False (True to be provided when reading from multiple sources)

What is being run:

We have calculated patch similarity for prev and buggy+nl with k values 1 and 10. 
