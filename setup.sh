conda create -n "PatchSearch" python=3.8
eval "$(conda shell.bash hook)"
conda activate PatchSearch
pip install -r requirements.txt
gdown --id 1QZV7Wu-_XpXNLl0Qv8_k22dVDG03Pjf9
unzip Patch-Dataset.zip
rm Patch-Dataset.zip